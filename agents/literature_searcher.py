"""
Literature Searcher agent.

Executes search queries via academic APIs and evaluates coverage.
"""

from __future__ import annotations

import config
from agents.base import BaseAgent
from state import PaperState
from tools.search import search_all


class LiteratureSearcher(BaseAgent):
    def __init__(self, client, model: str):
        prompt = config.load_prompt("literature_searcher").format(
            topic=config.TOPIC,
            topic_en=config.TOPIC_EN,
            focus_areas=", ".join(config.FOCUS_AREAS),
        )
        super().__init__("LiteratureSearcher", client, model, prompt)

    def run(self, state: PaperState, queries: list[str] | None = None) -> PaperState:
        """
        Search for papers using queries from state or provided list.

        If `queries` is given, only those are searched (supplementary round).
        Otherwise, uses state.search_queries.
        """
        to_search = queries if queries is not None else state.search_queries
        if not to_search:
            state.add_log(self.name, "No queries to search")
            return state

        state.add_log(self.name, f"Searching {len(to_search)} queries...")

        new_papers = search_all(to_search, papers_per_query=config.PAPERS_PER_QUERY)

        # Merge with existing papers (avoid duplicates)
        existing_ids = {p.paper_id for p in state.papers}
        added = 0
        for p in new_papers:
            if p.paper_id not in existing_ids:
                state.papers.append(p)
                existing_ids.add(p.paper_id)
                added += 1

        state.search_log.append(
            f"Searched {len(to_search)} queries → {len(new_papers)} results, "
            f"{added} new papers added (total: {state.paper_count()})"
        )
        state.add_log(
            self.name,
            f"Added {added} new papers (total: {state.paper_count()})",
        )
        return state
