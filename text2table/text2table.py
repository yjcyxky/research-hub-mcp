from __future__ import annotations

"""
text2table
----------

Pipeline for converting free-form text into Markdown tables by combining entity
recognition (GLiNER) and table-focused generation with Qwen.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from gliner import GLiNER
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_USER_PROMPT = "Based on the above information, output a Markdown table with the labels as headers."


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


def _infer_device_from_map(model: "AutoModelForCausalLM") -> "torch.device":
    import torch

    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        device_value = next(iter(model.hf_device_map.values()))
        if isinstance(device_value, str):
            return torch.device(device_value)
    if hasattr(model, "device"):
        return model.device  # type: ignore[arg-type]
    return torch.device("cpu")


@dataclass
class Text2Table:
    """Convert text to a Markdown table using GLiNER entities and a Qwen LLM."""

    labels: Sequence[str]
    gliner_model_name: str = "Ihor/gliner-biomed-large-v1.0"
    qwen_model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    threshold: float = 0.5
    cache_dir: Optional[Path] = None
    trust_remote_code: bool = True
    device: Optional[str] = None
    enable_thinking: bool = False

    _gliner: Optional["GLiNER"] = field(init=False, default=None)
    _tokenizer: Optional["AutoTokenizer"] = field(init=False, default=None)
    _llm: Optional["AutoModelForCausalLM"] = field(init=False, default=None)
    _inference_device: Optional["Any"] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.labels = _normalize_labels(self.labels)
        if not 0 < self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

    def _load_gliner(self) -> "GLiNER":
        from gliner import GLiNER

        logger.info("Loading GLiNER model: %s", self.gliner_model_name)
        kwargs: Dict[str, object] = {}
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)
        self._gliner = GLiNER.from_pretrained(self.gliner_model_name, **kwargs)
        if self.device:
            try:
                self._gliner.to(self.device)
            except Exception:
                logger.warning("Unable to move GLiNER model to device %s", self.device)
        return self._gliner

    def _load_qwen(self) -> Tuple["AutoTokenizer", "AutoModelForCausalLM"]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Qwen model: %s", self.qwen_model_name)
        kwargs: Dict[str, object] = {"trust_remote_code": self.trust_remote_code}
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)

        tokenizer = AutoTokenizer.from_pretrained(self.qwen_model_name, **kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map: Optional[str] = "auto" if target_device != "cpu" else None

        model = AutoModelForCausalLM.from_pretrained(
            self.qwen_model_name,
            torch_dtype=dtype,
            device_map=device_map,
            **kwargs,
        )
        if device_map is None and target_device:
            model.to(target_device)
        model.eval()

        inference_device = _infer_device_from_map(model)
        self._tokenizer = tokenizer
        self._llm = model
        self._inference_device = inference_device
        logger.info("Qwen inference device: %s", inference_device)
        return tokenizer, model

    @property
    def gliner(self) -> GLiNER:
        if self._gliner is None:
            self._load_gliner()
        return self._gliner  # type: ignore[return-value]

    @property
    def tokenizer(self) -> "AutoTokenizer":
        if self._tokenizer is None:
            self._load_qwen()
        return self._tokenizer  # type: ignore[return-value]

    @property
    def llm(self) -> "AutoModelForCausalLM":
        if self._llm is None:
            self._load_qwen()
        return self._llm  # type: ignore[return-value]

    def extract_entities(self, text: str) -> List[Dict[str, object]]:
        logger.debug("Running GLiNER with threshold=%s", self.threshold)
        entities = self.gliner.predict_entities(text, list(self.labels), threshold=self.threshold)
        entities = sorted(entities, key=lambda e: (e.get("start", 0), -(e.get("score", 0) or 0)))
        return entities

    def _format_entities(self, entities: List[Dict[str, object]]) -> str:
        if not entities:
            return "- No entities were found with the current threshold."

        grouped: Dict[str, List[str]] = {label: [] for label in self.labels}
        for entity in entities:
            label = str(entity.get("label", "")).strip()
            text = str(entity.get("text", "")).strip()
            if not label or not text:
                continue
            if label not in grouped:
                grouped[label] = []
            if text not in grouped[label]:
                grouped[label].append(text)

        lines: List[str] = []
        for label in self.labels:
            values = grouped.get(label, [])
            value_str = "; ".join(values) if values else "N/A"
            lines.append(f"- {label}: {value_str}")
        return "\n".join(lines)

    def build_prompt(
        self,
        text: str,
        entities: List[Dict[str, object]],
        user_prompt: Optional[str] = None,
    ) -> str:
        entity_block = self._format_entities(entities)
        header = ", ".join(self.labels)
        instruction = user_prompt.strip() if user_prompt else DEFAULT_USER_PROMPT
        
        if self.enable_thinking:
            prompt = (
                "You are a structured information extraction assistant. "
                "Use the extracted entities to fill a Markdown table.\n"

                f"Table headers (keep order): {header}\n"

                "Rules: one entity per row; if a field has multiple values, choose the single "
                "best value for the cell and put discarded/extra values into a Notes column; "
                "fill missing fields with N/A; append a Notes column after the provided headers.\n"

                f"User instruction (style/formatting hints): {instruction}\n\n"

                "Extracted entities:\n"
                f"{entity_block}\n\n"

                "Source text:\n"
                f"{text.strip()}\n\n"

                "IMPORTANT: Before outputting the final table, you must first think through "
                "the problem step by step. Use <think>...</think> tags to wrap your thinking process. "
                "After your thinking, output the final Markdown table.\n"
                "Format:\n"
                "<think>\n"
                "Your reasoning process here...\n"
                "</think>\n"
                "Then output the final Markdown table."
            )
        else:
            prompt = (
                "You are a structured information extraction assistant. "
                "Use the extracted entities to fill a Markdown table.\n"

                f"Table headers (keep order): {header}\n"

                "Rules: one entity per row; if a field has multiple values, choose the single "
                "best value for the cell and put discarded/extra values into a Notes column; "
                "fill missing fields with N/A; append a Notes column after the provided headers; "
                "output only the final Markdown table.\n"

                f"User instruction (style/formatting hints): {instruction}\n\n"

                "Extracted entities:\n"
                f"{entity_block}\n\n"

                "Source text:\n"
                f"{text.strip()}"
            )
        return prompt

    def generate_table(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        import torch

        tokenizer = self.tokenizer
        model = self.llm

        if self.enable_thinking:
            system_content = (
                "You convert extracted entities into concise Markdown tables. "
                "Use the provided headers in order, append a Notes column for extra values, "
                "one entity per row. Before outputting the final table, think through the "
                "problem step by step using <think>...</think> tags, then output the final table."
            )
            # Increase max_new_tokens to accommodate thinking content
            effective_max_tokens = max(max_new_tokens, 1024)
        else:
            system_content = (
                "You convert extracted entities into concise Markdown tables. "
                "Use the provided headers in order, append a Notes column for extra values, "
                "one entity per row, and return only the final table."
            )
            effective_max_tokens = max_new_tokens

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt")
        if self._inference_device:
            inputs = {k: v.to(self._inference_device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=effective_max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = output[0][inputs["input_ids"].shape[-1] :]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text.strip()
    
    def _parse_thinking_output(self, output: str) -> Tuple[str, str]:
        """Parse output to separate thinking and final result.
        
        Returns:
            Tuple of (thinking_text, final_result)
        """
        thinking_start = output.find("<think>")
        thinking_end = output.find("</think>")
        
        if thinking_start != -1 and thinking_end != -1:
            # Extract thinking content (skip the <think> tag, which is 7 characters)
            thinking_text = output[thinking_start + 7 : thinking_end].strip()
            # Extract final result (skip the </think> tag, which is 8 characters)
            final_result = output[thinking_end + 8 :].strip()
            return thinking_text, final_result
        else:
            # No thinking tags found, return empty thinking and full output as result
            return "", output

    def run(
        self,
        text: str,
        user_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Tuple[str, List[Dict[str, object]]]:
        """Run the text2table pipeline.
        
        Returns:
            Tuple of (table, entities). If thinking mode is enabled, table will contain
            both thinking and final result separated by thinking tags.
        """
        entities = self.extract_entities(text)
        prompt = self.build_prompt(text, entities, user_prompt=user_prompt)
        output = self.generate_table(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        if self.enable_thinking:
            thinking, table = self._parse_thinking_output(output)
            if thinking:
                logger.info("Thinking process:\n%s", thinking)
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
