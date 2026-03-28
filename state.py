"""
Shared state object for inter-agent communication.

All agents read from and write to a single PaperState instance.
No message queues — just a mutable dataclass passed by reference.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    PLANNING = "planning"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    WRITING = "writing"
    REVIEWING = "reviewing"
    DONE = "done"


@dataclass
class Paper:
    """A single academic paper collected from search APIs."""

    paper_id: str
    title: str
    authors: list[str]
    year: int
    abstract: str
    citation_count: int
    url: str
    source: str  # "semantic_scholar" | "arxiv"

    def short_ref(self) -> str:
        """e.g. 'Zhang et al., 2024'"""
        if not self.authors:
            return f"Unknown, {self.year}"
        first = self.authors[0].split()[-1]  # last name
        if len(self.authors) > 1:
            return f"{first} et al., {self.year}"
        return f"{first}, {self.year}"


@dataclass
class ThemeCluster:
    """A thematic group of related papers identified by the Analyst."""

    theme_name: str
    theme_name_zh: str
    description: str
    paper_ids: list[str]
    key_findings: list[str]
    research_gaps: list[str]


@dataclass
class PaperSection:
    """A section of the survey paper drafted by the Writer."""

    section_id: str        # e.g. "intro", "theme_1", "conclusion"
    title: str
    title_zh: str
    content: str           # markdown content
    status: str = "pending"  # "pending" | "drafted" | "revised"


@dataclass
class ReviewComment:
    """Structured feedback from the Reviewer on a paper section."""

    section_id: str
    comment: str
    severity: str          # "critical" | "major" | "minor"
    suggestion: str = ""


@dataclass
class PaperState:
    """
    Central state shared by all agents.

    Each agent reads the fields it needs and writes its output fields.
    The orchestrator (main.py) controls phase transitions.
    """

    # ── Topic Definition ─────────────────────────────────────────────
    topic: str = ""
    topic_en: str = ""
    focus_areas: list[str] = field(default_factory=list)

    # ── Pipeline Phase ───────────────────────────────────────────────
    current_phase: Phase = Phase.PLANNING

    # ── Search (written by LiteratureSearcher) ───────────────────────
    search_queries: list[str] = field(default_factory=list)
    papers: list[Paper] = field(default_factory=list)
    search_log: list[str] = field(default_factory=list)

    # ── Analysis (written by Analyst) ────────────────────────────────
    themes: list[ThemeCluster] = field(default_factory=list)
    taxonomy: str = ""
    timeline_summary: str = ""

    # ── Outline (written by ResearchLead) ────────────────────────────
    outline: list[dict] = field(default_factory=list)
    # Each entry: {"id": "intro", "title": "...", "title_zh": "...",
    #              "description": "...", "assigned_themes": [...]}

    # ── Written Sections (written by Writer) ─────────────────────────
    sections: list[PaperSection] = field(default_factory=list)

    # ── Review (written by Reviewer) ─────────────────────────────────
    review_comments: list[ReviewComment] = field(default_factory=list)

    # ── Activity Log ─────────────────────────────────────────────────
    log: list[str] = field(default_factory=list)

    # ── Helpers ──────────────────────────────────────────────────────

    def add_log(self, agent_name: str, message: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {agent_name}: {message}"
        self.log.append(entry)
        print(entry)

    def get_papers_by_ids(self, ids: list[str]) -> list[Paper]:
        id_set = set(ids)
        return [p for p in self.papers if p.paper_id in id_set]

    def get_section(self, section_id: str) -> PaperSection | None:
        for s in self.sections:
            if s.section_id == section_id:
                return s
        return None

    def paper_count(self) -> int:
        return len(self.papers)

    def theme_names(self) -> list[str]:
        return [t.theme_name for t in self.themes]
