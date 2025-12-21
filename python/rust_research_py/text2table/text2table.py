"""
text2table
----------

Pipeline for converting free-form text into TSV tables by combining
optional entity recognition (GLiNER service) and table-focused generation
through a vLLM OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

from .client import (
    AsyncGLiNERClient,
    AsyncVLLMClient,
    GLiNERClient,
    RetryPolicy,
    VLLMClient,
)
from .prompts import (
    SYSTEM_MESSAGE_DEFAULT,
    SYSTEM_MESSAGE_WITH_THINKING,
    DEFAULT_USER_INSTRUCTION,
    ROW_VALIDATION_SYSTEM_MESSAGE,
    build_entity_extraction_prompt,
    build_row_validation_prompt,
    format_entities_as_list,
    format_entities_with_relations,
)

if TYPE_CHECKING:  # pragma: no cover
    from gliner import GLiNER

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Single item to process in a batch."""

    text: str
    id: Optional[object] = None
    index: Optional[int] = None
    metadata: Optional[Dict[str, object]] = None


@dataclass
class BatchResult:
    """Result of processing a single batch item."""

    index: int
    text: str
    table: str
    entities: List[Dict[str, object]]
    thinking: str = ""
    id: Optional[object] = None
    metadata: Optional[Dict[str, object]] = None
    status: str = "ok"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        structured_table = (
            _structure_from_tsv(self.table, self.entities)
            if self.table
            else {
                "headers": [],
                "rows": [],
                "raw_table": self.table,
                "raw_table_with_think": self.table,
                "gliner_entities": self.entities or [],
            }
        )
        return {
            "index": self.index,
            "id": self.id,
            "text": self.text,
            "table": self.table,
            "pred_table": structured_table,
            "entities": self.entities,
            "thinking": self.thinking,
            "metadata": self.metadata,
            "status": self.status,
            "error": self.error,
        }


def _strip_think_blocks(table_text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    if not table_text:
        return table_text
    cleaned = table_text
    while "<think>" in cleaned and "</think>" in cleaned:
        start = cleaned.find("<think>")
        end = cleaned.find("</think>", start)
        if end == -1:
            break
        cleaned = cleaned[:start] + cleaned[end + len("</think>") :]
    return cleaned.strip()


def _parse_tsv_table(table_tsv: str) -> Tuple[List[str], List[List[str]]]:
    """Parse a TSV table into headers and rows."""
    if not table_tsv:
        return [], []
    table_tsv = _strip_think_blocks(table_tsv)
    lines = [line.strip() for line in table_tsv.splitlines() if line.strip()]
    if not lines:
        return [], []
    headers = [h.strip() for h in lines[0].split("\t")]
    rows: List[List[str]] = []
    for line in lines[1:]:
        rows.append([c.strip() for c in line.split("\t")])
    return headers, rows


def _rebuild_tsv_table(headers: List[str], rows: List[List[str]]) -> str:
    """Rebuild a TSV table from headers and rows."""
    if not headers:
        return ""
    header_line = "\t".join(headers)
    if not rows:
        return header_line
    row_lines = []
    confidence_idx = None
    for i, h in enumerate(headers):
        if h.lower() == "confidence":
            confidence_idx = i
            break
    for r in rows:
        padded = r + [""] * (len(headers) - len(r))
        if confidence_idx is not None and confidence_idx < len(headers):
            if not padded[confidence_idx]:
                padded[confidence_idx] = "1.0"
        row_lines.append("\t".join(padded[: len(headers)]))
    return "\n".join([header_line] + row_lines)


def _extract_row_from_text(headers: List[str], text: str, original_row: List[str]) -> List[str]:
    """Try to extract a corrected row from model output (TSV expected)."""
    parsed_headers, parsed_rows = _parse_tsv_table(text)
    if parsed_rows:
        # If headers length matches or model omitted headers, accept first row
        if not parsed_headers or len(parsed_headers) == len(headers):
            return parsed_rows[0]
    # Fallback: keep original
    return original_row


def _structure_from_tsv(table_tsv: str, gliner_entities: Optional[List[Dict[str, object]]] = None) -> Dict[str, object]:
    """Convert TSV text into a structured table dict."""
    raw_clean = _strip_think_blocks(table_tsv)
    headers, rows = _parse_tsv_table(raw_clean)
    structured_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        cells = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
        structured_rows.append({"cells": cells, "row_idx": idx})
    return {
        "headers": headers,
        "rows": structured_rows,
        "raw_table": raw_clean,
        "raw_table_with_think": table_tsv,
        "gliner_entities": gliner_entities or [],
    }


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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


@dataclass
class Text2Table:
    """Convert text to a TSV table using GLiNER entities and a vLLM service."""

    labels: Sequence[str]
    gliner_model_name: Optional[str] = "Ihor/gliner-biomed-large-v1.0"
    model_name: Optional[str] = None  # If None, use server default
    threshold: float = 0.5
    gliner_cache_dir: Optional[Path] = None
    gliner_device: Optional[str] = None
    enable_thinking: bool = False
    max_reasoning_tokens: Optional[int] = 2048
    include_confidence: bool = True
    server_url: Optional[str] = None  # vLLM server URL (required to run)
    gliner_url: Optional[str] = None  # GLiNER service URL (required when use_gliner is True)
    use_gliner: bool = True
    gliner_soft_threshold: Optional[float] = None  # Lower bound to pull extra GLiNER candidates as low-confidence hints
    enable_row_validation: bool = False  # Drop rows not supported by source text
    row_validation_mode: str = "substring"
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
        if self.gliner_soft_threshold is None:
            self.gliner_soft_threshold = _env_float(
                "TEXT2TABLE_GLINER_SOFT_THRESHOLD", None
            )
        if self.gliner_soft_threshold is not None and not 0 < self.gliner_soft_threshold <= 1:
            raise ValueError("gliner_soft_threshold must be between 0 and 1.")
        # Default lower bound cannot exceed primary threshold
        if (
            self.gliner_soft_threshold is not None
            and self.gliner_soft_threshold > self.threshold
        ):
            logger.warning(
                "gliner_soft_threshold (%.3f) is higher than threshold (%.3f); using the primary threshold instead",
                self.gliner_soft_threshold,
                self.threshold,
            )
            self.gliner_soft_threshold = self.threshold

        # Row validation toggle from env if not explicitly set
        self.enable_row_validation = _env_flag(
            "TEXT2TABLE_ENABLE_ROW_VALIDATION", self.enable_row_validation
        )
        self.row_validation_mode = (
            os.getenv("TEXT2TABLE_ROW_VALIDATION_MODE", self.row_validation_mode).strip()
            if os.getenv("TEXT2TABLE_ROW_VALIDATION_MODE")
            else self.row_validation_mode
        )
        if self.row_validation_mode not in {"substring", "llm"}:
            raise ValueError("Unsupported row_validation_mode. Use 'substring' or 'llm'.")
        self.include_confidence = _env_flag(
            "TEXT2TABLE_INCLUDE_CONFIDENCE", self.include_confidence
        )

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
            threshold_to_use = (
                self.gliner_soft_threshold if self.gliner_soft_threshold else self.threshold
            )
            entities = client.extract_entities(
                text, self.labels, threshold_to_use, model=self.gliner_model_name
            )
        else:
            threshold_to_use = self.gliner_soft_threshold or self.threshold
            entities = self._extract_entities_local(text, threshold_to_use)

        entities = self._tag_low_confidence(entities)
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

    def _extract_entities_local(self, text: str, threshold: float) -> List[Dict[str, object]]:
        logger.debug("Running local GLiNER with threshold=%s", threshold)
        try:
            entities = self.gliner.predict_entities(
                text, list(self.labels), threshold=threshold
            )
        except Exception as exc:
            logger.error("Local GLiNER failed: %s", exc)
            raise
        return entities

    def _tag_low_confidence(self, entities: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Mark low-confidence entities that fell below the main threshold but above soft threshold."""
        if self.gliner_soft_threshold is None:
            return entities
        tagged: List[Dict[str, object]] = []
        for e in entities:
            score = e.get("score")
            needs_verification = False
            try:
                if score is not None and float(score) < self.threshold:
                    needs_verification = True
            except Exception:
                needs_verification = False
            new_e = dict(e)
            if needs_verification:
                new_e["needs_verification"] = True
            tagged.append(new_e)
        return tagged

    def _validate_table_rows(self, source_text: str, table_markdown: str) -> str:
        """Validate rows via configured mode."""
        if not table_markdown:
            return table_markdown

        headers, rows = _parse_tsv_table(table_markdown)
        if not headers or not rows:
            return table_markdown

        if self.row_validation_mode == "substring":
            source_lower = source_text.lower()
            valid_rows: List[List[str]] = []
            for row in rows:
                ok = True
                for idx, cell in enumerate(row):
                    if idx < len(headers) and headers[idx].lower() == "confidence":
                        continue
                    cell_clean = cell.strip()
                    if not cell_clean or cell_clean.lower() == "n/a":
                        continue
                    if cell_clean.lower() not in source_lower:
                        ok = False
                        break
                if ok:
                    valid_rows.append(row)
            if not valid_rows:
                logger.info(
                    "Row validation dropped all rows (%d -> 0); returning header-only table.",
                    len(rows),
                )
            else:
                dropped = len(rows) - len(valid_rows)
                if dropped > 0:
                    logger.info(
                        "Row validation dropped %d row(s); kept %d.", dropped, len(valid_rows)
                    )
            return _rebuild_tsv_table(headers, valid_rows)

        if self.row_validation_mode == "llm":
            kept = self._llm_validate_rows(source_text, headers, rows)
            return _rebuild_tsv_table(headers, kept)

        return table_markdown

    async def _validate_table_rows_async(
        self, source_text: str, table_markdown: str
    ) -> str:
        """Async row validation dispatcher."""
        if not table_markdown:
            return table_markdown

        headers, rows = _parse_tsv_table(table_markdown)
        if not headers or not rows:
            return table_markdown

        if self.row_validation_mode == "substring":
            return self._validate_table_rows(source_text, table_markdown)
        if self.row_validation_mode == "llm":
            kept = await self._llm_validate_rows_async(source_text, headers, rows)
            return _rebuild_tsv_table(headers, kept)
        return table_markdown

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
        headers_list = list(self.labels)
        if self.include_confidence and "confidence" not in [h.lower() for h in headers_list]:
            headers_list.append("confidence")
        header = ", ".join(headers_list)

        return build_entity_extraction_prompt(
            text=text,
            entity_block=entity_block,
            headers=header,
            user_instruction=user_prompt,
            enable_thinking=self.enable_thinking,
            use_gliner=self.use_gliner,
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
            model=self.model_name,
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

    def _llm_validate_rows(
        self, source_text: str, headers: List[str], rows: List[List[str]]
    ) -> List[List[str]]:
        """Use the LLM to validate each row; drop unsupported rows."""
        client = self._get_vllm_client()
        kept: List[List[str]] = []
        for row in rows:
            prompt = build_row_validation_prompt(source_text, headers, row)
            messages = [
                {"role": "system", "content": ROW_VALIDATION_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ]
            try:
                resp = client.generate(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=128,
                    temperature=0.0,
                    top_p=0.1,
                )
            except Exception as exc:
                logger.error("Row validation request failed: %s", exc, exc_info=True)
                continue
            decision_text = resp.lower() if resp else ""
            keep = "decision: keep" in decision_text
            if not keep:
                continue
            corrected = _extract_row_from_text(headers, resp, row)
            kept.append(corrected)
        return kept

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
            if self.enable_row_validation:
                table = self._validate_table_rows(text, table)
            return table, entities
        else:
            if self.enable_row_validation:
                output = self._validate_table_rows(text, output)
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
        table = self._validate_table_rows(text, table) if self.enable_row_validation else table
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


@dataclass
class AsyncText2Table(Text2Table):
    """Async version of the Text2Table pipeline using AsyncOpenAI/httpx clients."""

    def __post_init__(self) -> None:
        super().__post_init__()

    async def _get_vllm_client(self) -> Any:
        if not self.server_url:
            raise ValueError("server_url is required. Local mode has been removed.")
        if self._vllm_client is None:
            self._vllm_client = AsyncVLLMClient(
                base_url=self.server_url,
                api_key=self.api_key,
                timeout=self.request_timeout,
                pool_size=self.pool_size,
                retry_policy=self.retry_policy,
            )
        return self._vllm_client

    async def _get_gliner_client(self) -> Any:
        if self._gliner_client is None:
            if not self.gliner_url:
                raise ValueError(
                    "gliner_url is required when use_gliner=True. "
                    "Set TEXT2TABLE_GLINER_URL or provide gliner_url explicitly."
                )
            self._gliner_client = AsyncGLiNERClient(
                base_url=self.gliner_url,
                api_key=self.gliner_api_key,
                timeout=self.request_timeout,
                pool_size=self.pool_size,
                retry_policy=self.retry_policy,
                default_model=self.gliner_model_name,
            )
        return self._gliner_client

    async def extract_entities(self, text: str) -> List[Dict[str, object]]:
        if not self.use_gliner:
            logger.info("GLiNER extraction disabled; skipping entity extraction.")
            return []
        if self.gliner_url:
            client = await self._get_gliner_client()
            threshold_to_use = (
                self.gliner_soft_threshold if self.gliner_soft_threshold else self.threshold
            )
            entities = await client.extract_entities(
                text, self.labels, threshold_to_use, model=self.gliner_model_name
            )
        else:
            loop = asyncio.get_running_loop()
            threshold_to_use = self.gliner_soft_threshold or self.threshold
            entities = await loop.run_in_executor(
                None, self._extract_entities_local, text, threshold_to_use
            )
        entities = self._tag_low_confidence(entities)
        entities = sorted(
            entities,
            key=lambda e: (e.get("start", 0), -(e.get("score", 0) or 0)),
        )
        return entities

    async def generate_table(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        reasoning_tokens: Optional[int] = None

        if self.enable_thinking:
            system_content = SYSTEM_MESSAGE_WITH_THINKING
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

        client = await self._get_vllm_client()
        logger.debug(
            "Calling vLLM (async) with max_tokens=%d, temperature=%.2f, top_p=%.2f",
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
        generated_text = await client.generate(
            messages=messages,
            model=self.model_name,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_reasoning_tokens=reasoning_tokens,
        )
        logger.info("Received text from vLLM (async) length=%d", len(generated_text))
        if generated_text:
            logger.info("First 500 chars: %s", repr(generated_text[:500]))
        else:
            logger.warning("Generated text is empty!")
        return generated_text.strip() if generated_text else ""

    async def _llm_validate_rows_async(
        self, source_text: str, headers: List[str], rows: List[List[str]]
    ) -> List[List[str]]:
        """Use the async LLM client to validate each row; drop unsupported rows."""
        client = await self._get_vllm_client()
        kept: List[List[str]] = []
        for row in rows:
            prompt = build_row_validation_prompt(source_text, headers, row)
            messages = [
                {"role": "system", "content": ROW_VALIDATION_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ]
            try:
                resp = await client.generate(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=128,
                    temperature=0.0,
                    top_p=0.1,
                )
            except Exception as exc:
                logger.error("Row validation request failed (async): %s", exc, exc_info=True)
                continue
            decision_text = resp.lower() if resp else ""
            keep = "decision: keep" in decision_text
            if not keep:
                continue
            corrected = _extract_row_from_text(headers, resp, row)
            kept.append(corrected)
        return kept

    async def run(
        self,
        text: str,
        user_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Tuple[str, List[Dict[str, object]]]:
        """Run the text2table pipeline asynchronously."""
        entities = await self.extract_entities(text)
        prompt = self.build_prompt(text, entities, user_prompt=user_prompt)
        output = await self.generate_table(
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
            if self.enable_row_validation:
                table = await self._validate_table_rows_async(text, table)
            return table, entities
        else:
            if self.enable_row_validation:
                output = await self._validate_table_rows_async(text, output)
            return output, entities

    async def run_with_thinking(
        self,
        text: str,
        user_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Tuple[str, str, List[Dict[str, object]]]:
        """Run the text2table pipeline asynchronously with thinking mode enabled."""
        entities = await self.extract_entities(text)
        prompt = self.build_prompt(text, entities, user_prompt=user_prompt)
        output = await self.generate_table(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        thinking, table = self._parse_thinking_output(output)
        if self.enable_row_validation:
            table = await self._validate_table_rows_async(text, table)
        return thinking, table, entities

    async def close(self) -> None:
        """Close underlying async HTTP clients to release connections."""
        if self._gliner_client:
            try:
                await self._gliner_client.close()
            except Exception:
                pass
        if self._vllm_client:
            try:
                await self._vllm_client.close()
            except Exception:
                pass

    async def run_many(
        self,
        items: Sequence[BatchItem],
        user_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        concurrency: int = 4,
        on_result: Optional[Callable[[BatchResult], Union[None, Awaitable[None]]]] = None,
    ) -> List[BatchResult]:
        """Process multiple items concurrently with bounded concurrency.

        Args:
            items: Sequence of BatchItem describing texts to process.
            user_prompt: Optional custom prompt appended to generation.
            max_new_tokens: Max generation tokens.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            concurrency: Maximum number of in-flight tasks.
            on_result: Optional async callback invoked per completed result.
        """
        if concurrency <= 0:
            raise ValueError("concurrency must be a positive integer.")

        normalized_items: List[BatchItem] = []
        for idx, item in enumerate(items):
            if not isinstance(item, BatchItem):
                raise TypeError("items must be a sequence of BatchItem")
            normalized_items.append(
                BatchItem(
                    text=item.text,
                    id=item.id,
                    metadata=item.metadata,
                    index=item.index if item.index is not None else idx,
                )
            )

        semaphore = asyncio.Semaphore(concurrency)
        results: List[BatchResult] = []

        async def _process(batch_item: BatchItem) -> BatchResult:
            async with semaphore:
                try:
                    if self.enable_thinking:
                        thinking, table, entities = await self.run_with_thinking(
                            batch_item.text,
                            user_prompt=user_prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    else:
                        table, entities = await self.run(
                            batch_item.text,
                            user_prompt=user_prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        thinking = ""
                    return BatchResult(
                        index=int(batch_item.index or 0),
                        id=batch_item.id,
                        metadata=batch_item.metadata,
                        text=batch_item.text,
                        table=table,
                        entities=entities,
                        thinking=thinking,
                        status="ok",
                        error=None,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to process batch item %s (index %s): %s",
                        batch_item.id,
                        batch_item.index,
                        exc,
                        exc_info=True,
                    )
                    return BatchResult(
                        index=int(batch_item.index or 0),
                        id=batch_item.id,
                        metadata=batch_item.metadata,
                        text=batch_item.text,
                        table="",
                        entities=[],
                        thinking="",
                        status="error",
                        error=str(exc),
                    )

        tasks = [asyncio.create_task(_process(item)) for item in normalized_items]

        async def _maybe_handle(result: BatchResult) -> None:
            if on_result:
                maybe = on_result(result)
                if asyncio.iscoroutine(maybe):
                    await maybe

        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            await _maybe_handle(res)

        # Preserve input order by index
        results.sort(key=lambda r: r.index)
        return results


def _load_batch_labels(
    labels: Sequence[str], labels_file: Optional[Union[str, Path]]
) -> List[str]:
    combined: List[str] = [label.strip() for label in labels if label and label.strip()]
    if labels_file:
        path = Path(labels_file)
        file_labels = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        combined.extend(file_labels)
    if not combined:
        raise ValueError("At least one label is required.")
    return combined


def _format_record(headers: Sequence[str], record: Sequence[str]) -> str:
    parts: List[str] = []
    for idx, field in enumerate(record):
        if idx >= len(headers):
            break
        parts.append(f"{headers[idx]}: {field}")
    return "\n".join(parts)


def run_text2table_batch(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    labels: Sequence[str],
    labels_file: Optional[Union[str, Path]] = None,
    prompt: Optional[str] = None,
    threshold: float = 0.5,
    gliner_model: str = "Ihor/gliner-biomed-large-v1.0",
    gliner_soft_threshold: Optional[float] = None,
    model_name: Optional[str] = None,
    enable_thinking: bool = False,
    server_url: Optional[str] = None,
    gliner_url: Optional[str] = None,
    disable_gliner: bool = False,
    enable_row_validation: bool = False,
    row_validation_mode: str = "substring",
    api_key: Optional[str] = None,
    gliner_api_key: Optional[str] = None,
    concurrency: int = 4,
) -> int:
    """Run text2table over a TSV/CSV file and write JSONL results."""
    if concurrency <= 0:
        raise ValueError("concurrency must be a positive integer.")

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(output_file)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    label_list = _load_batch_labels(labels, labels_file)
    if not server_url:
        raise ValueError("server_url is required for batch processing.")

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = [row for row in reader if row]

    if not rows:
        output_path.touch(exist_ok=True)
        return 0

    headers = rows[0]
    formatted_texts = [_format_record(headers, record) for record in rows[1:]]

    extractor = AsyncText2Table(
        labels=label_list,
        gliner_model_name=gliner_model,
        model_name=model_name,
        threshold=threshold,
        gliner_soft_threshold=gliner_soft_threshold,
        enable_thinking=enable_thinking,
        server_url=server_url,
        gliner_url=gliner_url,
        use_gliner=not disable_gliner,
        api_key=api_key or "dummy-key",
        gliner_api_key=gliner_api_key,
        enable_row_validation=enable_row_validation,
        row_validation_mode=row_validation_mode,
    )

    batch_items = [
        BatchItem(text=text_value, index=idx) for idx, text_value in enumerate(formatted_texts)
    ]

    async def _run_all() -> List[BatchResult]:
        try:
            return await extractor.run_many(
                batch_items,
                user_prompt=prompt,
                concurrency=concurrency,
            )
        finally:
            await extractor.close()

    results = asyncio.run(_run_all())

    with output_path.open("a", encoding="utf-8") as f:
        for result in results:
            original_text = (
                formatted_texts[result.index]
                if 0 <= result.index < len(formatted_texts)
                else ""
            )
            success = result.status == "ok"
            payload = {
                "original_text": original_text,
                "success": success,
                "table": result.table if success else None,
                "entities": result.entities if success else None,
                "thinking": result.thinking or None,
                "error": result.error if not success else None,
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")

    return len(results)
