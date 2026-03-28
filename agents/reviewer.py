"""
Reviewer agent.

Evaluates drafted sections and provides structured feedback.
"""

from __future__ import annotations

import config
from agents.base import BaseAgent
from state import PaperState, ReviewComment


class Reviewer(BaseAgent):
    def __init__(self, client, model: str):
        prompt = config.load_prompt("reviewer").format(
            topic=config.TOPIC,
            topic_en=config.TOPIC_EN,
        )
        super().__init__("Reviewer", client, model, prompt)

    def run(self, state: PaperState) -> PaperState:
        """Review all drafted sections and produce structured feedback."""
        state.add_log(self.name, f"Reviewing {len(state.sections)} sections...")
        state.review_comments = []

        for section in state.sections:
            state.add_log(self.name, f"  Reviewing: {section.title}")

            # Find the theme data assigned to this section
            section_def = next(
                (s for s in state.outline if s["id"] == section.section_id),
                {},
            )
            assigned_themes = section_def.get("assigned_themes", [])
            theme_context = ""
            for theme in state.themes:
                if theme.theme_name in assigned_themes:
                    theme_context += (
                        f"\n- {theme.theme_name}: {theme.description}\n"
                        f"  Key findings: {'; '.join(theme.key_findings)}\n"
                        f"  Papers: {len(theme.paper_ids)} assigned\n"
                    )

            request = f"""
Review the following section of a survey paper on "{state.topic}".

## Section: {section.title} / {section.title_zh}
{section.content}

## Underlying Theme Data (ground truth):
{theme_context if theme_context else "(This is a structural section like intro/conclusion)"}

Evaluate this section for:
1. Coverage of key findings from assigned themes
2. Proper use of citations
3. Logical flow and coherence
4. Academic tone and rigor

Respond with a JSON array of review comments:
[{{"section_id": "{section.section_id}", "comment": "...", "severity": "critical|major|minor", "suggestion": "..."}}]

If the section is satisfactory, return an empty array: []
"""
            result = self._call_llm_json(
                request,
                temperature=config.TEMPERATURE_STRUCTURED,
            )

            comments = result if isinstance(result, list) else []
            for c in comments:
                state.review_comments.append(ReviewComment(
                    section_id=c.get("section_id", section.section_id),
                    comment=c.get("comment", ""),
                    severity=c.get("severity", "minor"),
                    suggestion=c.get("suggestion", ""),
                ))

        total = len(state.review_comments)
        critical = sum(1 for c in state.review_comments if c.severity == "critical")
        major = sum(1 for c in state.review_comments if c.severity == "major")
        state.add_log(
            self.name,
            f"Review complete: {total} comments ({critical} critical, {major} major)",
        )
        return state
