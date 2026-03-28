"""
Survey Agent configuration.

To customize for a specific research topic, modify the TOPIC section below.
All other settings are topic-agnostic defaults.
"""

import os

# ── TOPIC (the only section you need to change per survey) ──────────────

TOPIC = os.environ.get("SURVEY_TOPIC", "Your Research Topic Here")
TOPIC_EN = os.environ.get("SURVEY_TOPIC_EN", "Your Research Topic Here (English)")
FOCUS_AREAS: list[str] = [
    # Add 3-6 focus areas / sub-topics, e.g.:
    # "deep learning",
    # "large language models",
]

# ── MODEL ────────────────────────────────────────────────────────────────

MODEL = os.environ.get("SURVEY_MODEL", "claude-sonnet-4-20250514")

# ── SEARCH ───────────────────────────────────────────────────────────────

PAPERS_PER_QUERY = int(os.environ.get("SURVEY_PAPERS_PER_QUERY", "20"))
MAX_SEARCH_ROUNDS = int(os.environ.get("SURVEY_MAX_SEARCH_ROUNDS", "2"))

# ── LLM PARAMETERS ──────────────────────────────────────────────────────

TEMPERATURE_STRUCTURED = 0.3   # for JSON / structured output tasks
TEMPERATURE_CREATIVE = 0.5     # for prose writing

MAX_TOKENS_DEFAULT = 4096
MAX_TOKENS_WRITING = 8192      # longer budget for paper sections

# ── OUTPUT ───────────────────────────────────────────────────────────────

OUTPUT_DIR = os.environ.get("SURVEY_OUTPUT_DIR", "output")

# ── PROMPTS ──────────────────────────────────────────────────────────────

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(agent_name: str) -> str:
    """Load a prompt template from prompts/ directory."""
    path = os.path.join(PROMPTS_DIR, f"{agent_name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
