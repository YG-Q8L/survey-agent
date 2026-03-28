"""
Research Lead agent — the orchestrator.

Responsibilities: planning queries, designing outline, quality gates, triage.
"""

from __future__ import annotations

import config
from agents.base import BaseAgent
from state import PaperState


class ResearchLead(BaseAgent):
    def __init__(self, client, model: str):
        prompt = config.load_prompt("research_lead").format(
            topic=config.TOPIC,
            topic_en=config.TOPIC_EN,
            focus_areas=", ".join(config.FOCUS_AREAS),
        )
        super().__init__("ResearchLead", client, model, prompt)

    # ── Phase 1: Planning ────────────────────────────────────────────

    def run(self, state: PaperState) -> PaperState:
        """Generate search queries and initial paper outline."""
        state.add_log(self.name, "Generating search queries and outline...")

        request = f"""
Based on the survey topic and focus areas, please generate:

1. "queries": a list of 8-12 search queries (mix of English and Chinese)
   that together cover the full scope of this survey.
2. "outline": a list of paper sections, each with:
   - "id": short identifier (e.g. "intro", "background", "theme_1", ...)
   - "title": section title in English
   - "title_zh": section title in Chinese
   - "description": what this section should cover (1-2 sentences)

Topic: {state.topic} / {state.topic_en}
Focus areas: {', '.join(state.focus_areas)}

Respond with a single JSON object containing "queries" and "outline".
"""
        result = self._call_llm_json(
            request,
            max_tokens=config.MAX_TOKENS_DEFAULT,
            temperature=config.TEMPERATURE_STRUCTURED,
        )
        state.search_queries = result.get("queries", [])
        state.outline = result.get("outline", [])

        state.add_log(
            self.name,
            f"Generated {len(state.search_queries)} queries, "
            f"{len(state.outline)} outline sections",
        )
        return state

    # ── Phase 2b: Search Quality Gate ────────────────────────────────

    def review_search_coverage(self, state: PaperState) -> list[str]:
        """
        Review search results and return additional queries if gaps found.
        Returns an empty list if coverage is sufficient.
        """
        state.add_log(self.name, "Reviewing search coverage...")

        paper_summary = "\n".join(
            f"- [{p.year}] {p.title} (citations: {p.citation_count})"
            for p in state.papers[:80]  # cap to fit context
        )

        request = f"""
We searched for papers on "{state.topic}" and found {state.paper_count()} papers.

Here are the papers found:
{paper_summary}

Original search queries used:
{chr(10).join(f'- {q}' for q in state.search_queries)}

Focus areas we need to cover: {', '.join(state.focus_areas)}

Evaluate coverage:
1. Are all focus areas well-represented?
2. Are there obvious gaps in sub-topics?
3. Do we have enough recent papers (last 3 years)?

If coverage is sufficient, respond: {{"additional_queries": []}}
If gaps exist, respond: {{"additional_queries": ["query1", "query2", ...], "reason": "..."}}
"""
        result = self._call_llm_json(
            request,
            temperature=config.TEMPERATURE_STRUCTURED,
        )
        additional = result.get("additional_queries", [])
        if additional:
            state.add_log(self.name, f"Found gaps, adding {len(additional)} queries")
        else:
            state.add_log(self.name, "Coverage is sufficient")
        return additional

    # ── Phase 3b: Refine Outline ─────────────────────────────────────

    def refine_outline(self, state: PaperState) -> PaperState:
        """Map analyzed themes to paper sections, refine outline."""
        state.add_log(self.name, "Refining outline with theme data...")

        themes_desc = "\n".join(
            f"- {t.theme_name} / {t.theme_name_zh}: {t.description} "
            f"({len(t.paper_ids)} papers)"
            for t in state.themes
        )

        request = f"""
We have analyzed the collected papers and identified these themes:
{themes_desc}

Current outline:
{_format_outline(state.outline)}

Please refine the outline by mapping themes to sections. Update the outline so that:
1. Each thematic section is connected to one or more themes.
2. Sections are ordered for a logical narrative flow.
3. Each section entry includes "assigned_themes" (list of theme names).
4. Keep intro, background, discussion/future directions, and conclusion sections.

Respond with a JSON object: {{"outline": [updated section list]}}
Each section: {{"id": "...", "title": "...", "title_zh": "...", "description": "...", "assigned_themes": [...]}}
"""
        result = self._call_llm_json(
            request,
            temperature=config.TEMPERATURE_STRUCTURED,
        )
        state.outline = result.get("outline", state.outline)
        state.add_log(self.name, f"Outline refined: {len(state.outline)} sections")
        return state

    # ── Phase 5b: Triage Reviews ─────────────────────────────────────

    def triage_reviews(self, state: PaperState) -> list[str]:
        """Return section IDs that need revision (critical + major issues)."""
        sections_to_revise: set[str] = set()
        for comment in state.review_comments:
            if comment.severity in ("critical", "major"):
                sections_to_revise.add(comment.section_id)

        if sections_to_revise:
            state.add_log(
                self.name,
                f"Sections needing revision: {', '.join(sections_to_revise)}",
            )
        else:
            state.add_log(self.name, "No critical/major issues found")
        return list(sections_to_revise)


def _format_outline(outline: list[dict]) -> str:
    lines = []
    for s in outline:
        themes = s.get("assigned_themes", [])
        theme_str = f" [themes: {', '.join(themes)}]" if themes else ""
        lines.append(f"- {s['id']}: {s.get('title', '')} / {s.get('title_zh', '')}"
                      f"  — {s.get('description', '')}{theme_str}")
    return "\n".join(lines)
