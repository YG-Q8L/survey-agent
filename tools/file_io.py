"""
File I/O tools for state persistence and paper output.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from state import PaperState


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def save_state_snapshot(
    state: PaperState,
    filename: str = "state_snapshot.json",
    output_dir: str = "output",
) -> str:
    """Save current state to JSON for debugging and crash recovery."""
    ensure_output_dir(output_dir)
    data = {
        "topic": state.topic,
        "topic_en": state.topic_en,
        "current_phase": state.current_phase.value,
        "search_queries": state.search_queries,
        "papers": [asdict(p) for p in state.papers],
        "themes": [asdict(t) for t in state.themes],
        "outline": state.outline,
        "sections": [asdict(s) for s in state.sections],
        "review_comments": [asdict(c) for c in state.review_comments],
        "log": state.log,
    }
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def save_markdown_paper(
    state: PaperState,
    filename: str = "survey_paper.md",
    output_dir: str = "output",
) -> str:
    """Assemble all sections into a final markdown paper."""
    ensure_output_dir(output_dir)
    lines: list[str] = []

    # Title
    lines.append(f"# {state.topic_en}")
    if state.topic and state.topic != state.topic_en:
        lines.append(f"# {state.topic}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sections
    for section in state.sections:
        lines.append(section.content)
        lines.append("")

    # References
    lines.append("---")
    lines.append("")
    lines.append("## References")
    lines.append("")
    for i, paper in enumerate(state.papers, 1):
        authors_str = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += " et al."
        lines.append(
            f"[{i}] {authors_str}. \"{paper.title}\". {paper.year}. {paper.url}"
        )
    lines.append("")

    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
