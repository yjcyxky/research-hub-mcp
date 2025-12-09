"""Tests for the prompts module."""

import pytest

from text2table.prompts import (
    SYSTEM_MESSAGE_DEFAULT,
    SYSTEM_MESSAGE_WITH_THINKING,
    DEFAULT_USER_INSTRUCTION,
    build_entity_extraction_prompt,
    format_entities_as_list,
    format_entities_with_relations,
)


class TestSystemMessages:
    """Test system message constants."""

    def test_system_message_default_exists(self):
        assert SYSTEM_MESSAGE_DEFAULT
        assert "TSV table" in SYSTEM_MESSAGE_DEFAULT

    def test_system_message_with_thinking_exists(self):
        assert SYSTEM_MESSAGE_WITH_THINKING
        assert "<think>" in SYSTEM_MESSAGE_WITH_THINKING

    def test_default_user_instruction_exists(self):
        assert DEFAULT_USER_INSTRUCTION
        assert "TSV table" in DEFAULT_USER_INSTRUCTION


class TestFormatEntitiesAsList:
    """Test format_entities_as_list function."""

    def test_disabled_gliner(self):
        result = format_entities_as_list([], ["drug", "ade"], use_gliner=False)
        assert "Entity extraction disabled" in result

    def test_no_entities(self):
        result = format_entities_as_list([], ["drug", "ade"], use_gliner=True)
        assert "No entities were found" in result

    def test_single_entity(self):
        entities = [{"label": "drug", "text": "aspirin"}]
        result = format_entities_as_list(entities, ["drug", "ade"])
        assert "drug: aspirin" in result
        assert "ade: N/A" in result

    def test_multiple_entities_same_label(self):
        entities = [
            {"label": "drug", "text": "aspirin"},
            {"label": "drug", "text": "ibuprofen"},
        ]
        result = format_entities_as_list(entities, ["drug", "ade"])
        assert "aspirin; ibuprofen" in result

    def test_deduplicates_entities(self):
        entities = [
            {"label": "drug", "text": "aspirin"},
            {"label": "drug", "text": "aspirin"},  # Duplicate
        ]
        result = format_entities_as_list(entities, ["drug"])
        # Should only appear once
        assert result.count("aspirin") == 1


class TestFormatEntitiesWithRelations:
    """Test format_entities_with_relations function."""

    def test_disabled_gliner(self):
        result = format_entities_with_relations([], ["drug", "ade"], use_gliner=False)
        assert "Entity extraction disabled" in result

    def test_no_entities(self):
        result = format_entities_with_relations([], ["drug", "ade"], use_gliner=True)
        assert "No entities were found" in result

    def test_single_entity_per_label(self):
        entities = [
            {"label": "drug", "text": "aspirin"},
            {"label": "ade", "text": "headache"},
        ]
        result = format_entities_with_relations(entities, ["drug", "ade"])
        assert "drug: aspirin" in result
        assert "ade: headache" in result

    def test_multiple_entities_listed_separately(self):
        """Multiple entities of the same label should be on separate lines."""
        entities = [
            {"label": "drug", "text": "aspirin"},
            {"label": "drug", "text": "ibuprofen"},
        ]
        result = format_entities_with_relations(entities, ["drug", "ade"])
        # Each should be on its own line
        assert "- drug: aspirin" in result
        assert "- drug: ibuprofen" in result

    def test_multi_entity_hint_added(self):
        """When multiple entities found, a hint should be added."""
        entities = [
            {"label": "drug", "text": "aspirin"},
            {"label": "drug", "text": "ibuprofen"},
            {"label": "ade", "text": "headache"},
        ]
        result = format_entities_with_relations(entities, ["drug", "ade"])
        assert "2 drug(s)" in result
        assert "separate row for each" in result

    def test_no_hint_for_single_entities(self):
        """No hint when each label has only one entity."""
        entities = [
            {"label": "drug", "text": "aspirin"},
            {"label": "ade", "text": "headache"},
        ]
        result = format_entities_with_relations(entities, ["drug", "ade"])
        assert "separate row" not in result


class TestBuildEntityExtractionPrompt:
    """Test build_entity_extraction_prompt function."""

    def test_basic_prompt_structure(self):
        prompt = build_entity_extraction_prompt(
            text="Patient took aspirin.",
            entity_block="- drug: aspirin",
            headers="drug, ade",
        )
        assert "drug, ade" in prompt
        assert "aspirin" in prompt
        assert "Patient took aspirin." in prompt

    def test_custom_user_instruction(self):
        prompt = build_entity_extraction_prompt(
            text="Text",
            entity_block="- drug: test",
            headers="drug, ade",
            user_instruction="Custom instruction here",
        )
        assert "Custom instruction here" in prompt

    def test_default_instruction_when_none(self):
        prompt = build_entity_extraction_prompt(
            text="Text",
            entity_block="- drug: test",
            headers="drug, ade",
            user_instruction=None,
        )
        assert DEFAULT_USER_INSTRUCTION in prompt

    def test_thinking_mode_includes_tags(self):
        prompt = build_entity_extraction_prompt(
            text="Text",
            entity_block="- drug: test",
            headers="drug, ade",
            enable_thinking=True,
        )
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "think through the problem" in prompt

    def test_non_thinking_mode_no_tags(self):
        prompt = build_entity_extraction_prompt(
            text="Text",
            entity_block="- drug: test",
            headers="drug, ade",
            enable_thinking=False,
        )
        assert "<think>" not in prompt
        assert "</think>" not in prompt
        assert "only the final TSV table" in prompt

    def test_multi_row_rules_present(self):
        """Prompt should contain rules about generating multiple rows."""
        prompt = build_entity_extraction_prompt(
            text="Text",
            entity_block="- drug: test",
            headers="drug, ade",
        )
        assert "ONE ROW per unique entity relation" in prompt
        assert "TWO rows" in prompt  # Example in rules

    def test_notes_column_guidance(self):
        """Prompt should contain guidance about Notes column usage."""
        prompt = build_entity_extraction_prompt(
            text="Text",
            entity_block="- drug: test",
            headers="drug, ade",
        )
        assert "Notes column is ONLY for" in prompt
        assert "Do NOT put other valid entities in Notes" in prompt

    def test_soft_hint_guidance_included(self):
        """GLiNER outputs should be treated as soft hints, not hard constraints."""
        prompt = build_entity_extraction_prompt(
            text="Text",
            entity_block="- drug: test",
            headers="drug, ade",
            use_gliner=True,
        )
        assert "SOFT hints" in prompt
        assert "Prefer the source text" in prompt

    def test_prompt_without_gliner_uses_generic_guidance(self):
        prompt = build_entity_extraction_prompt(
            text="Text",
            entity_block="- Entity extraction disabled",
            headers="drug, ade",
            use_gliner=False,
        )
        assert "No upstream entity hints" in prompt
        assert "GLiNER soft hints" not in prompt
