"""
Prompt templates for text2table.

All prompts are centralized here for easy maintenance and modification.
"""

from typing import List, Optional


# =============================================================================
# System Messages
# =============================================================================

SYSTEM_MESSAGE_DEFAULT = (
    "You convert extracted entities into concise Markdown tables. "
    "Use the provided headers in order, append a Notes column only when there is "
    "genuinely ambiguous or uncertain information. "
    "Generate one row per unique relation (e.g., drug-ADE pair, drug-drug interaction, etc.)."
)

SYSTEM_MESSAGE_WITH_THINKING = (
    "You convert extracted entities into concise Markdown tables. "
    "Use the provided headers in order, append a Notes column only when there is "
    "genuinely ambiguous or uncertain information. "
    "Generate one row per unique relation (e.g., drug-ADE pair, drug-drug interaction, etc.). "
    "Before outputting the final table, think through the problem step by step "
    "using <think>...</think> tags, then output the final table."
)


# =============================================================================
# User Prompt Templates
# =============================================================================

DEFAULT_USER_INSTRUCTION = (
    "Based on the above information, output a Markdown table with the labels as headers."
)


def build_entity_extraction_prompt(
    text: str,
    entity_block: str,
    headers: str,
    user_instruction: Optional[str] = None,
    enable_thinking: bool = False,
) -> str:
    """
    Build the main prompt for entity extraction and table generation.

    Args:
        text: The source text to extract entities from.
        entity_block: Formatted string of extracted entities.
        headers: Comma-separated list of column headers.
        user_instruction: Optional custom instruction (defaults to DEFAULT_USER_INSTRUCTION).
        enable_thinking: Whether to include thinking mode instructions.

    Returns:
        The complete prompt string.
    """
    instruction = (user_instruction.strip() if user_instruction else DEFAULT_USER_INSTRUCTION)

    # Core extraction rules - emphasize generating multiple rows
    rules = (
        "Rules:\n"
        "1. Generate ONE ROW per unique entity relation. For example:\n"
        "   - If drug A and drug B both cause ADE X, output TWO rows: (A, X) and (B, X)\n"
        "   - If drug A causes ADE X and ADE Y, output TWO rows: (A, X) and (A, Y)\n"
        "   - Combine: if drugs A, B cause ADEs X, Y → output FOUR rows: (A,X), (A,Y), (B,X), (B,Y)\n"
        "2. The Notes column is ONLY for:\n"
        "   - Genuinely uncertain or ambiguous information\n"
        "   - Additional context that doesn't fit in other columns\n"
        "   - Do NOT put other valid entities in Notes; create separate rows instead\n"
        "3. Fill missing fields with N/A"
    )

    if enable_thinking:
        prompt = (
            "You are a structured information extraction assistant. "
            "Use the extracted entities to fill a Markdown table.\n\n"
            f"Table headers (keep order): {headers}\n\n"
            f"{rules}\n\n"
            f"User instruction: {instruction}\n\n"
            "Extracted entities or instructions:\n"
            f"{entity_block}\n\n"
            "Source text:\n"
            f"{text.strip()}\n\n"
            "IMPORTANT: Before outputting the final table, think through the problem step by step. "
            "Use <think>...</think> tags to wrap your reasoning process. "
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
            "Use the extracted entities to fill a Markdown table.\n\n"
            f"Table headers (keep order): {headers}\n\n"
            f"{rules}\n\n"
            "Output only the final Markdown table.\n\n"
            f"User instruction: {instruction}\n\n"
            "Extracted entities or instructions:\n"
            f"{entity_block}\n\n"
            "Source text:\n"
            f"{text.strip()}"
        )

    return prompt


# =============================================================================
# Entity Formatting Templates
# =============================================================================

def format_entities_as_list(
    entities: List[dict],
    labels: List[str],
    use_gliner: bool = True,
) -> str:
    """
    Format extracted entities as a bulleted list for the prompt.

    Args:
        entities: List of entity dictionaries with 'label' and 'text' keys.
        labels: List of expected labels.
        use_gliner: Whether GLiNER extraction is enabled.

    Returns:
        Formatted string of entities.
    """
    if not use_gliner:
        return "- Entity extraction disabled; infer entities directly from the source text."

    if not entities:
        return "- No entities were found with the current threshold."

    # Group entities by label
    grouped: dict[str, List[str]] = {label: [] for label in labels}
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
    for label in labels:
        values = grouped.get(label, [])
        value_str = "; ".join(values) if values else "N/A"
        lines.append(f"- {label}: {value_str}")

    return "\n".join(lines)


# =============================================================================
# Relationship-Aware Entity Formatting (for multi-row generation)
# =============================================================================

def format_entities_with_relations(
    entities: List[dict],
    labels: List[str],
    use_gliner: bool = True,
) -> str:
    """
    Format extracted entities with explicit relationship hints.

    This format helps the model understand that multiple combinations
    of entities should result in multiple rows.

    Args:
        entities: List of entity dictionaries.
        labels: List of expected labels.
        use_gliner: Whether GLiNER extraction is enabled.

    Returns:
        Formatted string with relationship hints.
    """
    if not use_gliner:
        return "- Entity extraction disabled; infer entities directly from the source text."

    if not entities:
        return "- No entities were found with the current threshold."

    # Group entities by label
    grouped: dict[str, List[str]] = {label: [] for label in labels}
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
    for label in labels:
        values = grouped.get(label, [])
        if values:
            # List each value separately to emphasize they are distinct
            for v in values:
                lines.append(f"- {label}: {v}")
        else:
            lines.append(f"- {label}: N/A")

    # Add explicit count hint for multi-entity cases
    entity_counts = [(label, len(grouped.get(label, []))) for label in labels]
    multi_counts = [(l, c) for l, c in entity_counts if c > 1]

    if multi_counts:
        count_info = ", ".join([f"{c} {l}(s)" for l, c in multi_counts])
        lines.append(f"\nNote: Found {count_info}. Generate a separate row for each combination.")

    return "\n".join(lines)
