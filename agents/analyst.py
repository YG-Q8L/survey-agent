"""
Analyst agent.

Clusters collected papers into themes, identifies findings and gaps.
"""

from __future__ import annotations

import config
from agents.base import BaseAgent
from state import PaperState, ThemeCluster


class Analyst(BaseAgent):
    def __init__(self, client):
        prompt = config.load_prompt("analyst").format(
            topic=config.TOPIC,
            topic_en=config.TOPIC_EN,
            focus_areas=", ".join(config.FOCUS_AREAS),
        )
        super().__init__("Analyst", client, prompt)

    def run(self, state: PaperState) -> PaperState:
        """Cluster papers into themes and produce analysis."""
        state.add_log(self.name, f"Analyzing {state.paper_count()} papers...")

        # Format papers for the LLM (truncate abstracts to save tokens)
        paper_entries = []
        for p in state.papers:
            abstract_short = p.abstract[:250] + "..." if len(p.abstract) > 250 else p.abstract
            paper_entries.append(
                f"ID: {p.paper_id}\n"
                f"Title: {p.title}\n"
                f"Year: {p.year} | Citations: {p.citation_count}\n"
                f"Abstract: {abstract_short}"
            )
        papers_text = "\n---\n".join(paper_entries)

        # ── Step 1: Thematic clustering ──────────────────────────────
        cluster_request = f"""
Here are {state.paper_count()} papers collected for a survey on "{state.topic}":

{papers_text}

Cluster these papers into 4-7 thematic groups. For each theme, provide:
{{
  "theme_name": "English name",
  "theme_name_zh": "Chinese name",
  "description": "2-3 sentences about this research direction",
  "paper_ids": ["id1", "id2", ...],
  "key_findings": ["finding 1", "finding 2", ...],
  "research_gaps": ["gap 1", "gap 2", ...]
}}

A paper can belong to multiple themes if relevant.
Respond with: {{"themes": [list of theme objects]}}
"""
        cluster_result = self._call_llm_json(
            cluster_request,
            max_tokens=config.MAX_TOKENS_WRITING,
            temperature=config.TEMPERATURE_STRUCTURED,
        )

        state.themes = [
            ThemeCluster(**t) for t in cluster_result.get("themes", [])
        ]
        state.add_log(self.name, f"Identified {len(state.themes)} themes")

        # ── Step 2: Timeline and taxonomy ────────────────────────────
        timeline_request = f"""
Based on the {state.paper_count()} papers analyzed, provide:

1. "taxonomy": A high-level classification of the research landscape in this field
   (3-5 paragraphs describing the major categories and how they relate).
2. "timeline_summary": A chronological summary of how this field has developed,
   noting key milestones and shifts in research focus.

Respond with: {{"taxonomy": "...", "timeline_summary": "..."}}
"""
        meta_result = self._call_llm_json(
            timeline_request,
            max_tokens=config.MAX_TOKENS_DEFAULT,
            temperature=config.TEMPERATURE_STRUCTURED,
        )
        state.taxonomy = meta_result.get("taxonomy", "")
        state.timeline_summary = meta_result.get("timeline_summary", "")

        state.add_log(self.name, "Analysis complete (themes + taxonomy + timeline)")
        return state
