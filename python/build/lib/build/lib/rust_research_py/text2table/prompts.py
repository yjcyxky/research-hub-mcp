"""
Prompt templates for text2table.

All prompts are centralized here for easy maintenance and modification.
"""

from typing import List, Optional


# =============================================================================
# System Messages
# =============================================================================

SYSTEM_MESSAGE_DEFAULT = (
    "You convert source text (and optional candidate entities) into concise TSV tables. "
    "Use the provided headers in order, append a Notes column only when there is "
    "genuinely ambiguous or uncertain information. "
    "Candidate entities are SOFT hints - rely on the source text as ground truth, and correct, drop, or extend hints as needed. "
    "Generate one row per unique relation (e.g., drug-ADE pair, drug-drug interaction, etc.). "
    "Always include a confidence column (0-1, default 1.0) as the last column."
)

SYSTEM_MESSAGE_WITH_THINKING = (
    "You convert source text (and optional candidate entities) into concise TSV tables. "
    "Use the provided headers in order, append a Notes column only when there is "
    "genuinely ambiguous or uncertain information. "
    "Candidate entities are SOFT hints - rely on the source text as ground truth, and correct, drop, or extend hints as needed. "
    "Generate one row per unique relation (e.g., drug-ADE pair, drug-drug interaction, etc.). "
    "Always include a confidence column (0-1, default 1.0) as the last column. "
    "Before outputting the final table, think through the problem step by step "
    "using <think>...</think> tags, then output the final table."
)


# =============================================================================
# User Prompt Templates
# =============================================================================

DEFAULT_USER_INSTRUCTION = (
    "Based on the above information, output a TSV table with the labels as headers."
)


def build_entity_extraction_prompt(
    text: str,
    entity_block: str,
    headers: str,
    user_instruction: Optional[str] = None,
    enable_thinking: bool = False,
    use_gliner: bool = True,
) -> str:
    """
    Build the main prompt for entity extraction and table generation.

    Args:
        text: The source text to extract entities from.
        entity_block: Formatted string of extracted entities.
        headers: Comma-separated list of column headers.
        user_instruction: Optional custom instruction (defaults to DEFAULT_USER_INSTRUCTION).
        enable_thinking: Whether to include thinking mode instructions.
        use_gliner: Whether upstream GLiNER hints are being supplied.

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
        "3. Fill missing fields with N/A\n"
        "4. Always include a 'confidence' column (0-1, default 1.0) as the last column.\n"
    )

    if use_gliner:
        rules += (
            "5. GLiNER candidates are SOFT hints. Trust the source text as the ground truth: "
            "fix or drop any hint that is unsupported or mislabeled, and add entities/relations "
            "that appear in the text even if GLiNER missed them."
        )
    else:
        rules += (
            "5. No upstream entity hints are enforced. Extract entities/relations directly from the source text."
        )

    hint_guidance = (
        "Use the source text as the authority. Any candidate entities listed below come from an upstream tagger "
        "(e.g., GLiNER) and are fallible. Treat them as starting points only - correct labels/spans, discard "
        "unsupported hints, and add missing entities/relations based on the text."
    )
    hint_guidance_disabled = (
        "No upstream entity hints are provided. Perform full extraction from the source text."
    )
    hint_header = (
        "GLiNER soft hints (verify/correct/extend):"
        if use_gliner
        else "Extraction guidance:"
    )

    if enable_thinking:
        prompt = (
            "You are a structured information extraction assistant. "
            "The source text is the ground truth. "
            + (hint_guidance if use_gliner else hint_guidance_disabled)
            + "\n\n"
            f"Table headers (keep order): {headers}\n\n"
            f"{rules}\n"
            "Hint handling:\n"
            "- Prefer the source text over any hints.\n"
            "- Correct or discard hints that conflict with the text or label schema.\n"
            "- Add entities/relations present in the text even if absent from hints.\n"
            "- If no hints are present, still extract directly from the text.\n\n"
            f"User instruction: {instruction}\n\n"
            f"{hint_header}\n"
            f"{entity_block}\n\n"
            "Source text:\n"
            f"{text.strip()}\n\n"
            "IMPORTANT: Before outputting the final table, think through the problem step by step. "
            "Use <think>...</think> tags to wrap your reasoning process. "
            "After your thinking, output the final TSV table (tab-separated, header line first).\n"
            "Format:\n"
            "<think>\n"
            "Your reasoning process here...\n"
            "</think>\n"
            "Then output the final TSV table."
        )
    else:
        prompt = (
            "You are a structured information extraction assistant. "
            "The source text is the ground truth. "
            + (hint_guidance if use_gliner else hint_guidance_disabled)
            + "\n\n"
            f"Table headers (keep order): {headers}\n\n"
            f"{rules}\n"
            "Hint handling:\n"
            "- Prefer the source text over any hints.\n"
            "- Correct or discard hints that conflict with the text or label schema.\n"
            "- Add entities/relations present in the text even if absent from hints.\n"
            "- If no hints are present, still extract directly from the text.\n\n"
            "Output only the final TSV table (tab-separated, header line first).\n\n"
            f"User instruction: {instruction}\n\n"
            f"{hint_header}\n"
            f"{entity_block}\n\n"
            "Source text:\n"
            f"{text.strip()}"
        )

    return prompt


# =============================================================================
# Row Validation Prompt
# =============================================================================

ROW_VALIDATION_SYSTEM_MESSAGE = (
    "You are a strict fact checker for a drug-adverse event table. "
    "Given a source text and one candidate table row, decide if the row is supported by the text. "
    "Only rows explicitly supported by the text should be kept."
)


def build_row_validation_prompt(
    text: str,
    headers: List[str],
    row: List[str],
) -> str:
    """
    Build a prompt asking the model to validate a single TSV table row.

    The model should respond with:
    DECISION: KEEP or DECISION: DROP
    Optionally, if DROP but a correction exists, provide one corrected TSV row (tab-separated).
    """
    header_line = "\t".join(headers)
    row_values = "\t".join(row + [""] * (len(headers) - len(row)))
    return (
        "Source text:\n"
        f"{text.strip()}\n\n"
        f"Candidate row (TSV):\n"
        f"{header_line}\n"
        f"{row_values}\n\n"
        "Respond with:\n"
        "- 'DECISION: KEEP' if the row is fully supported by the source text.\n"
        "- 'DECISION: DROP' if the row is unsupported or incorrect.\n"
        "If you drop but can correct the row, add a corrected TSV row after your decision.\n"
        "Examples:\n"
        "DECISION: KEEP\n"
        "DECISION: DROP\n"
        "DECISION: DROP\n"
        "drug\tade\tconfidence\n"
        "aspirin\theadache\t1.0"
    )


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
        return (
            "- No entities were found with the current threshold. "
            "Infer entities and relations directly from the source text."
        )

    # Group entities by label
    grouped: dict[str, List[str]] = {label: [] for label in labels}
    for entity in entities:
        label = str(entity.get("label", "")).strip()
        text = str(entity.get("text", "")).strip()
        needs_verification = bool(entity.get("needs_verification"))
        if not label or not text:
            continue
        if label not in grouped:
            grouped[label] = []
        display = text
        if needs_verification:
            display = f"{text} (low confidence - verify)"
        if display not in grouped[label]:
            grouped[label].append(display)

    lines: List[str] = [
        "Candidate entities from GLiNER (soft hints - verify against the source text; "
        "fix labels/spans and add missing entities):"
    ]
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

    lines.append(
        "\nTreat these as suggestions only. Drop or correct any hint not supported by the text, "
        "and include entities/relations mentioned in the text even if they are missing above."
    )

    return "\n".join(lines)
