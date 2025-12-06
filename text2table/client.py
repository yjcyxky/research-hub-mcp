"""
Service clients for text2table.

This module provides HTTP clients for vLLM (OpenAI-compatible) and GLiNER
services with connection pooling and retry logic for production use.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class RetryPolicy:
    """Simple retry configuration with exponential backoff.

    attempts represents retry attempts after the initial request.
    """

    attempts: int = 3
    backoff_factor: float = 1.5
    max_backoff: float = 10.0

    def backoff(self, attempt: int) -> float:
        delay = self.backoff_factor * (2 ** (attempt - 1))
        return min(delay, self.max_backoff)


def _build_httpx_client(
    base_url: Optional[str],
    timeout: float,
    pool_size: int,
    headers: Optional[dict] = None,
) -> "httpx.Client":
    if httpx is None:
        raise ImportError("httpx is required. Install it with: pip install httpx")

    limits = httpx.Limits(
        max_connections=pool_size,
        max_keepalive_connections=pool_size,
    )
    transport = httpx.HTTPTransport(retries=0)
    return httpx.Client(
        base_url=base_url,
        timeout=timeout,
        limits=limits,
        transport=transport,
        headers=headers,
    )


class VLLMClient:
    """Client for communicating with a vLLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy-key",  # vLLM doesn't require real API key
        timeout: float = 300.0,
        pool_size: int = 10,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        """Initialize vLLM client.
        
        Args:
            base_url: Base URL of the vLLM server (default: http://localhost:8000/v1)
            api_key: API key (not used by vLLM, but required by OpenAI client)
            timeout: Request timeout in seconds
            pool_size: Connection pool size for keep-alive HTTP connections
            retry_policy: Optional RetryPolicy configuration
        """
        if OpenAI is None:
            raise ImportError(
                "OpenAI client is required. Install it with: pip install openai"
            )
        
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"

        self.retry_policy = retry_policy or RetryPolicy()
        self.available_models: List[str] = []
        self._http_client = _build_httpx_client(
            base_url=self.base_url, timeout=timeout, pool_size=pool_size
        )
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout,
            http_client=self._http_client,
        )
        
        # Get the actual model name(s) from the server
        self.model_name = self._get_model_name()
        logger.info("Initialized vLLM client for %s, using model: %s", self.base_url, self.model_name)

    def _execute_with_retry(self, func: Any, description: str) -> Any:
        last_exc: Optional[Exception] = None
        total_attempts = self.retry_policy.attempts + 1

        for attempt in range(1, total_attempts + 1):
            try:
                return func()
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                if attempt >= total_attempts:
                    break
                delay = self.retry_policy.backoff(attempt)
                logger.warning(
                    "%s failed (attempt %d/%d). Retrying in %.1fs. Error: %s",
                    description,
                    attempt,
                    total_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)

        if last_exc:
            logger.error("%s failed after %d attempts", description, total_attempts)
            raise last_exc
        raise RuntimeError(f"{description} failed with unknown error")
    
    def _get_model_name(self) -> str:
        """Get the model name from the vLLM server.
        
        Returns:
            The model name, or 'default' if unable to determine
        """
        try:
            models = self._execute_with_retry(
                lambda: self.client.models.list(), "List models from vLLM"
            )
            if models.data and len(models.data) > 0:
                self.available_models = [m.id for m in models.data if getattr(m, "id", None)]
                # vLLM typically returns the model name in the id field
                model_name = models.data[0].id
                logger.debug("Found model on server: %s", model_name)
                return model_name
            else:
                logger.warning("No models found on server, using 'default'")
                return "default"
        except Exception as e:
            logger.warning("Could not get model name from server: %s, using 'default'", e)
            return "default"

    def _model_or_default(self, model: Optional[str]) -> str:
        if model and self.available_models and model not in self.available_models:
            logger.warning(
                "Requested model '%s' not in server models %s; falling back to server default '%s'",
                model,
                self.available_models,
                self.model_name,
            )
            return self.model_name
        return model or self.model_name

    @staticmethod
    def _stringify_content(content: object) -> str:
        """Convert OpenAI/vLLM content/parts into a plain string."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_value = item.get("text")
                    if text_value:
                        parts.append(str(text_value))
                else:
                    text_value = getattr(item, "text", None)
                    if text_value:
                        parts.append(str(text_value))
            return "\n".join([p for p in parts if p])
        if isinstance(content, dict):
            text_value = content.get("text")
            return str(text_value) if text_value else ""
        text_value = getattr(content, "text", None)
        if text_value:
            return str(text_value)
        try:
            return str(content)
        except Exception:
            return ""

    @classmethod
    def _combine_message_content(cls, message: Any) -> str:
        """Combine content and reasoning_content, wrapping reasoning in <think> tags."""
        if message is None:
            return ""

        reasoning_text = cls._stringify_content(
            getattr(message, "reasoning_content", None)
        ).strip()
        final_text = cls._stringify_content(getattr(message, "content", None)).strip()

        parts: List[str] = []
        if reasoning_text:
            if "<think>" in reasoning_text and "</think>" in reasoning_text:
                parts.append(reasoning_text)
            else:
                parts.append(f"<think>\n{reasoning_text}\n</think>")
        if final_text:
            parts.append(final_text)

        return "\n".join(parts).strip()

    def generate(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        max_reasoning_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate text using the vLLM server.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (if None, uses the model detected from server)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Optional list of stop sequences
            max_reasoning_tokens: Optional limit for reasoning tokens (vLLM reasoning output)
            reasoning_effort: Optional reasoning effort (low/medium/high) if server supports it
            
        Returns:
            Generated text string
        """
        # Use detected model name if not specified; fall back if unknown
        model_to_use = self._model_or_default(model)
        
        try:
            extra_body: dict = {}
            if max_reasoning_tokens:
                extra_body["max_reasoning_tokens"] = max_reasoning_tokens
                # Some OpenAI-compatible servers expect max_output_tokens alongside reasoning
                extra_body.setdefault("max_output_tokens", max_tokens)
            if reasoning_effort:
                extra_body["reasoning"] = {"effort": reasoning_effort}

            call_kwargs = {}
            if extra_body:
                call_kwargs["extra_body"] = extra_body

            def _call() -> Any:
                return self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    **call_kwargs,
                )

            response = self._execute_with_retry(_call, "Call vLLM chat completion")
            
            if not response.choices:
                raise ValueError("No response from vLLM server")

            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message is None:
                raise ValueError("vLLM response missing message content")

            reasoning_text = self._stringify_content(
                getattr(message, "reasoning_content", None)
            ).strip()
            content_only = self._stringify_content(getattr(message, "content", None)).strip()
            content = self._combine_message_content(message)

            logger.info(
                "Received response from vLLM (content_len=%d, reasoning_len=%d, combined_len=%d)",
                len(content_only),
                len(reasoning_text),
                len(content),
            )

            # Check finish reason
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason:
                logger.info("Finish reason: %s", finish_reason)
                if finish_reason == "length":
                    logger.warning("Response was truncated due to max_tokens limit!")
            
            if not content:
                logger.warning("Empty response from vLLM server")
            else:
                # Log full content for debugging (especially for short responses)
                logger.info("Response content (repr): %s", repr(content))
                if len(content) < 100:
                    logger.warning("Response is very short! This might indicate a problem.")
            return content
        except Exception as e:
            logger.error("Error calling vLLM server: %s", e, exc_info=True)
            raise

    def health_check(self) -> bool:
        """Check if the vLLM server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Try to list models as a health check
            models = self._execute_with_retry(
                lambda: self.client.models.list(), "Health check (list models)"
            )
            return len(models.data) > 0
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False

    def close(self) -> None:
        """Close the underlying HTTP client."""
        try:
            self._http_client.close()
        except Exception:
            pass


class GLiNERClient:
    """Client for communicating with a GLiNER entity extraction service."""

    def __init__(
        self,
        base_url: str = "http://localhost:9001",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        pool_size: int = 10,
        retry_policy: Optional[RetryPolicy] = None,
        default_model: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.retry_policy = retry_policy or RetryPolicy()

        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = _build_httpx_client(
            base_url=self.base_url,
            timeout=timeout,
            pool_size=pool_size,
            headers=headers,
        )

    def _execute_with_retry(self, func: Any, description: str) -> Any:
        last_exc: Optional[Exception] = None
        total_attempts = self.retry_policy.attempts + 1

        for attempt in range(1, total_attempts + 1):
            try:
                return func()
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                if attempt >= total_attempts:
                    break
                delay = self.retry_policy.backoff(attempt)
                logger.warning(
                    "%s failed (attempt %d/%d). Retrying in %.1fs. Error: %s",
                    description,
                    attempt,
                    total_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)

        if last_exc:
            logger.error("%s failed after %d attempts", description, total_attempts)
            raise last_exc
        raise RuntimeError(f"{description} failed with unknown error")

    @staticmethod
    def _parse_entities(payload: Any) -> List[dict]:
        if isinstance(payload, list):
            return [entity for entity in payload if isinstance(entity, dict)]
        if isinstance(payload, dict):
            for key in ("entities", "data", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [entity for entity in value if isinstance(entity, dict)]
        logger.warning("GLiNER response did not contain entities list, returning empty list")
        return []

    def extract_entities(
        self,
        text: str,
        labels: Sequence[str],
        threshold: float,
        model: Optional[str] = None,
    ) -> List[dict]:
        payload = {
            "text": text,
            "labels": list(labels),
            "threshold": threshold,
        }
        model_to_use = model or self.default_model
        if model_to_use:
            payload["model"] = model_to_use

        def _call() -> Any:
            response = self._client.post("/extract", json=payload)
            response.raise_for_status()
            return response

        response = self._execute_with_retry(_call, "Call GLiNER extraction service")
        try:
            parsed = response.json()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("GLiNER response is not valid JSON: %s", exc)
            raise
        entities = self._parse_entities(parsed)
        logger.info("GLiNER service returned %d entities", len(entities))
        return entities

    def health_check(self) -> bool:
        try:
            response = self._client.get("/health")
            return response.status_code < 500
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("GLiNER health check failed: %s", exc)
            return False

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
