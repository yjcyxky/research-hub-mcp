"""Text-to-table extraction utilities."""

from text2table.text2table import Text2Table
from text2table.prompts import DEFAULT_USER_INSTRUCTION

# For backwards compatibility
DEFAULT_USER_PROMPT = DEFAULT_USER_INSTRUCTION

__all__ = ["Text2Table", "DEFAULT_USER_PROMPT", "DEFAULT_USER_INSTRUCTION"]
