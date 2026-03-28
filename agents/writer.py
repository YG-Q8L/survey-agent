"""
Writer agent.

Drafts and revises individual sections of the survey paper.
"""

from __future__ import annotations

import config
from agents.base import BaseAgent
from state import PaperState, PaperSection


class Writer(BaseAgent):
    def __init__(self, client, model: str):
        prompt = config.load_prompt("writer").format(
            topic=config.TOPIC,
            topic_en=config.TOPIC_EN,
        )
        super().__init__("Writer", client, model, prompt)

    def run(self, state: PaperState) -> PaperState:
        """Draft all sections defined in the outline."""
        state.add_log(self.name, f"Writing {len(state.outline)} sections...")
        state.sections = []

        for i, section_def in enumerate(state.outline):
            sid = section_def["id"]
            title = section_def.get("title", "")
            title_zh = section_def.get("title_zh", "")
            desc = section_def.get("description", "")
            assigned_themes = section_def.get("assigned_themes", [])

            state.add_log(self.name, f"  [{i + 1}/{len(state.outline)}] Writing: {title}")

            prompt = self._build_section_prompt(state, sid, title, title_zh, desc, assigned_themes)
            content = self._call_llm_once(
                prompt,
                max_tokens=config.MAX_TOKENS_WRITING,
                temperature=config.TEMPERATURE_CREATIVE,
            )

            state.sections.append(PaperSection(
                section_id=sid,
                title=title,
                title_zh=title_zh,
                content=content,
                status="drafted",
            ))

        state.add_log(self.name, f"Drafted {len(state.sections)} sections")
        return state

    def revise(self, state: PaperState, section_ids: list[str]) -> PaperState:
        """Revise specific sections based on reviewer feedback."""
        state.add_log(self.name, f"Revising {len(section_ids)} sections...")

        for sid in section_ids:
            section = state.get_section(sid)
            if not section:
                continue

            # Collect relevant review comments
            comments = [
                c for c in state.review_comments if c.section_id == sid
            ]
            feedback_text = "\n".join(
                f"- [{c.severity}] {c.comment} → Suggestion: {c.suggestion}"
                for c in comments
            )

            request = f"""
Please revise the following section based on reviewer feedback.

## Section: {section.title} / {section.title_zh}

### Current Draft:
{section.content}

### Reviewer Feedback:
{feedback_text}

Rewrite the section addressing all critical and major feedback.
Output the complete revised section in Markdown.
"""
            revised_content = self._call_llm_once(
                request,
                max_tokens=config.MAX_TOKENS_WRITING,
                temperature=config.TEMPERATURE_CREATIVE,
            )
            section.content = revised_content
            section.status = "revised"
            state.add_log(self.name, f"  Revised: {section.title}")

        return state

    def _build_section_prompt(
        self,
        state: PaperState,
        section_id: str,
        title: str,
        title_zh: str,
        description: str,
        assigned_themes: list[str],
    ) -> str:
        """Build a writing prompt tailored to the section type."""
        # Gather theme data for assigned themes
        theme_data = ""
        if assigned_themes:
            for theme in state.themes:
                if theme.theme_name in assigned_themes:
                    papers = state.get_papers_by_ids(theme.paper_ids)
                    paper_refs = "\n".join(
                        f"    - [{p.short_ref()}] {p.title}" for p in papers
                    )
                    theme_data += f"""
  Theme: {theme.theme_name} / {theme.theme_name_zh}
  Description: {theme.description}
  Key findings: {'; '.join(theme.key_findings)}
  Research gaps: {'; '.join(theme.research_gaps)}
  Papers:
{paper_refs}
"""

        # Section-type specific instructions
        if section_id == "intro":
            extra = (
                f"Write an introduction for this survey. "
                f"Total papers reviewed: {state.paper_count()}. "
                f"Themes covered: {', '.join(state.theme_names())}. "
                f"Provide motivation, scope, and a roadmap of the paper. "
                f"Target: 500-800 words."
            )
        elif section_id == "background":
            extra = (
                f"Write a background section covering the foundational concepts. "
                f"Include the development timeline:\n{state.timeline_summary}\n"
                f"Target: 600-1000 words."
            )
        elif section_id in ("discussion", "future_directions"):
            all_gaps = []
            for t in state.themes:
                all_gaps.extend(t.research_gaps)
            extra = (
                f"Write a discussion of challenges and future directions. "
                f"Research gaps identified across all themes:\n"
                + "\n".join(f"- {g}" for g in all_gaps)
                + "\nTarget: 600-1000 words."
            )
        elif section_id == "conclusion":
            extra = (
                f"Write a conclusion summarizing the survey's key takeaways. "
                f"Themes covered: {', '.join(state.theme_names())}. "
                f"Target: 300-500 words."
            )
        elif section_id == "abstract":
            extra = (
                f"Write an abstract for this survey paper. "
                f"Papers reviewed: {state.paper_count()}. "
                f"Themes: {', '.join(state.theme_names())}. "
                f"Target: 150-250 words."
            )
        else:
            extra = (
                f"Write this thematic section based on the assigned theme data below. "
                f"Target: 800-1200 words."
            )

        return f"""
## Section to Write

ID: {section_id}
Title: {title} / {title_zh}
Description: {description}

{extra}

{f"### Theme Data:{theme_data}" if theme_data else ""}

Write the section in Markdown. Start with the section header (## {title}).
"""
