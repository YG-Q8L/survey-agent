"""
Survey Agent configuration.

Topic-specific settings are loaded from topics/<name>.py via load_topic().
All other settings here are topic-agnostic defaults.
"""

import importlib
import os

# ── TOPIC (loaded dynamically from topics/) ─────────────────────────

TOPIC: str = ""
TOPIC_EN: str = ""
FOCUS_AREAS: list[str] = []


def load_topic(name: str) -> None:
    """
    Load topic config from topics/<name>.py into this module's globals.

    Usage:
        config.load_topic("ich")
        print(config.TOPIC)  # "非遗数字化保护"
    """
    global TOPIC, TOPIC_EN, FOCUS_AREAS
    module = importlib.import_module(f"topics.{name}")
    TOPIC = getattr(module, "TOPIC")
    TOPIC_EN = getattr(module, "TOPIC_EN")
    FOCUS_AREAS = getattr(module, "FOCUS_AREAS")


# ── MODEL ────────────────────────────────────────────────────────────

MODEL = os.environ.get("SURVEY_MODEL", "claude-sonnet-4-20250514")

# ── SEARCH ───────────────────────────────────────────────────────────

PAPERS_PER_QUERY = int(os.environ.get("SURVEY_PAPERS_PER_QUERY", "20"))
MAX_SEARCH_ROUNDS = int(os.environ.get("SURVEY_MAX_SEARCH_ROUNDS", "2"))

# ── LLM PARAMETERS ──────────────────────────────────────────────────

TEMPERATURE_STRUCTURED = 0.3   # for JSON / structured output tasks
TEMPERATURE_CREATIVE = 0.5     # for prose writing

MAX_TOKENS_DEFAULT = 4096
MAX_TOKENS_WRITING = 8192      # longer budget for paper sections

# ── OUTPUT ───────────────────────────────────────────────────────────

OUTPUT_DIR = os.environ.get("SURVEY_OUTPUT_DIR", "output")

# ── PROMPTS ──────────────────────────────────────────────────────────

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(agent_name: str) -> str:
    """Load a prompt template from prompts/ directory."""
    path = os.path.join(PROMPTS_DIR, f"{agent_name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
