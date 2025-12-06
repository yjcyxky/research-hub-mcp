"""
text2table
----------

Pipeline for converting free-form text into Markdown tables by combining
optional entity recognition (GLiNER service) and table-focused generation
through a vLLM OpenAI-compatible endpoint.
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from text2table.client import GLiNERClient, RetryPolicy, VLLMClient
from text2table.prompts import (
    SYSTEM_MESSAGE_DEFAULT,
    SYSTEM_MESSAGE_WITH_THINKING,
    DEFAULT_USER_INSTRUCTION,
    build_entity_extraction_prompt,
    format_entities_as_list,
    format_entities_with_relations,
)

if TYPE_CHECKING:  # pragma: no cover
    from gliner import GLiNER

logger = logging.getLogger(__name__)


def _normalize_labels(labels: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for label in labels:
        if label is None:
            continue
        normalized = label.strip()
        if not normalized:
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
    if not cleaned:
        raise ValueError("At least one non-empty label is required.")
    return cleaned


@dataclass
class Text2Table:
    """Convert text to a Markdown table using GLiNER entities and a vLLM service."""

    labels: Sequence[str]
    gliner_model_name: Optional[str] = "Ihor/gliner-biomed-large-v1.0"
    qwen_model_name: Optional[str] = None  # If None, use server default
    threshold: float = 0.5
    gliner_cache_dir: Optional[Path] = None
    gliner_device: Optional[str] = None
    enable_thinking: bool = False
    max_reasoning_tokens: Optional[int] = 2048
    server_url: Optional[str] = None  # vLLM server URL (required to run)
    gliner_url: Optional[str] = None  # GLiNER service URL (required when use_gliner is True)
    use_gliner: bool = True
    api_key: str = "dummy-key"
    gliner_api_key: Optional[str] = None
    request_timeout: float = 120.0
    pool_size: int = 10
    max_retries: int = 3
    backoff_factor: float = 1.5
    max_backoff: float = 10.0

    _vllm_client: Optional[Any] = field(init=False, default=None)
    _gliner_client: Optional[Any] = field(init=False, default=None)
    _gliner: Optional["GLiNER"] = field(init=False, default=None)
    _retry_policy: Optional[Any] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.labels = _normalize_labels(self.labels)
        if not 0 < self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        # Allow env vars to define service endpoints
        self.server_url = self.server_url or os.getenv("TEXT2TABLE_VLLM_URL")
        self.gliner_url = self.gliner_url or os.getenv("TEXT2TABLE_GLINER_URL")
        self._retry_policy = RetryPolicy(
            attempts=self.max_retries,
            backoff_factor=self.backoff_factor,
            max_backoff=self.max_backoff,
        )

    @property
    def retry_policy(self):
        return self._retry_policy

    def _get_vllm_client(self) -> "Any":
        if not self.server_url:
            raise ValueError("server_url is required. Local mode has been removed.")
        if self._vllm_client is None:
            self._vllm_client = VLLMClient(
                base_url=self.server_url,
                api_key=self.api_key,
                timeout=self.request_timeout,
                pool_size=self.pool_size,
                retry_policy=self.retry_policy,
            )
        return self._vllm_client

    def _get_gliner_client(self) -> "Any":
        if self._gliner_client is None:
            if not self.gliner_url:
                raise ValueError(
                    "gliner_url is required when use_gliner=True. "
                    "Set TEXT2TABLE_GLINER_URL or provide gliner_url explicitly."
                )
            self._gliner_client = GLiNERClient(
                base_url=self.gliner_url,
                api_key=self.gliner_api_key,
                timeout=self.request_timeout,
                pool_size=self.pool_size,
                retry_policy=self.retry_policy,
                default_model=self.gliner_model_name,
            )
        return self._gliner_client

    def extract_entities(self, text: str) -> List[Dict[str, object]]:
        if not self.use_gliner:
            logger.info("GLiNER extraction disabled; skipping entity extraction.")
            return []
        if self.gliner_url:
            client = self._get_gliner_client()
            entities = client.extract_entities(
                text, self.labels, self.threshold, model=self.gliner_model_name
            )
        else:
            entities = self._extract_entities_local(text)
        entities = sorted(
            entities,
            key=lambda e: (e.get("start", 0), -(e.get("score", 0) or 0)),
        )
        return entities

    def _load_gliner_local(self) -> "GLiNER":
        from gliner import GLiNER

        logger.info("Loading local GLiNER model: %s", self.gliner_model_name)
        kwargs: Dict[str, object] = {}
        if self.gliner_cache_dir:
            kwargs["cache_dir"] = str(self.gliner_cache_dir)
        self._gliner = GLiNER.from_pretrained(self.gliner_model_name, **kwargs)
        if self.gliner_device:
            try:
                self._gliner.to(self.gliner_device)
            except Exception:
                logger.warning(
                    "Unable to move GLiNER model to device %s", self.gliner_device
                )
        return self._gliner

    @property
    def gliner(self) -> "GLiNER":
        if self._gliner is None:
            self._load_gliner_local()
        return self._gliner  # type: ignore[return-value]

    def _extract_entities_local(self, text: str) -> List[Dict[str, object]]:
        logger.debug("Running local GLiNER with threshold=%s", self.threshold)
        try:
            entities = self.gliner.predict_entities(
                text, list(self.labels), threshold=self.threshold
            )
        except Exception as exc:
            logger.error("Local GLiNER failed: %s", exc)
            raise
        return entities

    def _format_entities(self, entities: List[Dict[str, object]]) -> str:
        """Format entities for the prompt using the prompts module."""
        return format_entities_with_relations(
            entities=entities,
            labels=list(self.labels),
            use_gliner=self.use_gliner,
        )

    def build_prompt(
        self,
        text: str,
        entities: List[Dict[str, object]],
        user_prompt: Optional[str] = None,
    ) -> str:
        """Build the prompt using the centralized prompts module."""
        entity_block = self._format_entities(entities)
        header = ", ".join(self.labels)

        return build_entity_extraction_prompt(
            text=text,
            entity_block=entity_block,
            headers=header,
            user_instruction=user_prompt,
            enable_thinking=self.enable_thinking,
        )

    def generate_table(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        reasoning_tokens: Optional[int] = None

        if self.enable_thinking:
            system_content = SYSTEM_MESSAGE_WITH_THINKING
            # Increase max_new_tokens to accommodate thinking content
            effective_max_tokens = max(max_new_tokens, 1024)
            reasoning_tokens = (
                self.max_reasoning_tokens
                if self.max_reasoning_tokens and self.max_reasoning_tokens > 0
                else max(effective_max_tokens * 2, 1024)
            )
        else:
            system_content = SYSTEM_MESSAGE_DEFAULT
            effective_max_tokens = max_new_tokens

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        client = self._get_vllm_client()
        logger.debug(
            "Calling vLLM with max_tokens=%d, temperature=%.2f, top_p=%.2f",
            effective_max_tokens,
            temperature,
            top_p,
        )
        if reasoning_tokens:
            logger.debug("Requesting reasoning tokens: %d", reasoning_tokens)
        logger.debug("System message length: %d", len(system_content))
        logger.debug("User message length: %d", len(prompt))
        logger.debug(
            "User message preview: %s",
            prompt[:200] if len(prompt) > 200 else prompt,
        )
        generated_text = client.generate(
            messages=messages,
            model=self.qwen_model_name,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_reasoning_tokens=reasoning_tokens,
        )
        logger.info("Received text from vLLM (length: %d)", len(generated_text))
        if generated_text:
            logger.info("First 500 chars: %s", repr(generated_text[:500]))
        else:
            logger.warning("Generated text is empty!")
        return generated_text.strip() if generated_text else ""

    def _parse_thinking_output(self, output: str) -> Tuple[str, str]:
        """Parse output to separate thinking and final result.
        
        Returns:
            Tuple of (thinking_text, final_result)
        """
        # Look for <think> tags (as specified in build_prompt)
        start_tag = "<think>"
        end_tag = "</think>"

        thinking_start = output.find(start_tag)
        thinking_end = output.find(end_tag)

        if thinking_start != -1 and thinking_end != -1 and thinking_end > thinking_start:
            # Extract thinking content (skip the start tag)
            thinking_text = output[thinking_start + len(start_tag) : thinking_end].strip()
            # Extract final result (skip the end tag)
            final_result = output[thinking_end + len(end_tag) :].strip()
            logger.debug(
                "Parsed thinking output: thinking_len=%d, result_len=%d",
                len(thinking_text),
                len(final_result),
            )
            return thinking_text, final_result
        else:
            # No thinking tags found, return empty thinking and full output as result
            logger.info(
                "No thinking tags found in output (length: %d), returning full output as result",
                len(output),
            )
            if output:
                logger.debug("First 500 chars of output: %s", output[:500])
            # Even if no tags found, return the full output as the result
            return "", output

    def run(
        self,
        text: str,
        user_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Tuple[str, List[Dict[str, object]]]:
        """Run the text2table pipeline in normal mode."""
        entities = self.extract_entities(text)
        prompt = self.build_prompt(text, entities, user_prompt=user_prompt)
        output = self.generate_table(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        logger.debug("Generated output length: %d", len(output))

        if self.enable_thinking:
            thinking, table = self._parse_thinking_output(output)
            if thinking:
                logger.info("Thinking process:\n%s", thinking)
            logger.debug("Final table length: %d", len(table))
            return table, entities
        else:
            return output, entities

    def run_with_thinking(
        self,
        text: str,
        user_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Tuple[str, str, List[Dict[str, object]]]:
        """Run the text2table pipeline with thinking mode enabled.
        
        Returns:
            Tuple of (thinking_text, table, entities)
        """
        entities = self.extract_entities(text)
        prompt = self.build_prompt(text, entities, user_prompt=user_prompt)
        output = self.generate_table(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        thinking, table = self._parse_thinking_output(output)
        return thinking, table, entities

    def close(self) -> None:
        """Close underlying HTTP clients to release connections."""
        if self._gliner_client:
            try:
                self._gliner_client.close()
            except Exception:
                pass
        if self._vllm_client:
            try:
                self._vllm_client.close()
            except Exception:
                pass
