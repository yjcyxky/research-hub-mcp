from types import SimpleNamespace

from text2table.client import VLLMClient


def test_combine_message_content_wraps_reasoning_and_content():
    message = SimpleNamespace(
        reasoning_content="step1\nstep2",
        content="| a | b |\n| --- | --- |",
    )

    combined = VLLMClient._combine_message_content(message)

    assert "<think>" in combined
    assert "step1" in combined
    assert combined.strip().endswith("| a | b |\n| --- | --- |")


def test_combine_message_content_preserves_existing_think_tags():
    message = SimpleNamespace(
        reasoning_content="<think>already wrapped</think>",
        content="final table",
    )

    combined = VLLMClient._combine_message_content(message)

    assert combined.count("<think>") == 1
    assert "already wrapped" in combined
    assert combined.endswith("final table")

