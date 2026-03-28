"""
Survey Agent — Multi-agent system for writing technical survey papers.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python main.py --topic ich
    python main.py --topic example
"""

import argparse

from anthropic import Anthropic

import config
from state import PaperState, Phase
from agents import ResearchLead, LiteratureSearcher, Analyst, Writer, Reviewer
from tools.file_io import save_state_snapshot, save_markdown_paper


def main():
    # ── Parse args ───────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Survey Agent")
    parser.add_argument(
        "--topic", required=True,
        help="Topic name — must match a file in topics/ (e.g. 'ich', 'example')",
    )
    args = parser.parse_args()

    config.load_topic(args.topic)

    # ── Initialize ───────────────────────────────────────────────────
    client = Anthropic()
    model = config.MODEL

    state = PaperState(
        topic=config.TOPIC,
        topic_en=config.TOPIC_EN,
        focus_areas=list(config.FOCUS_AREAS),
    )

    lead = ResearchLead(client=client, model=model)
    searcher = LiteratureSearcher(client=client, model=model)
    analyst = Analyst(client=client, model=model)
    writer = Writer(client=client, model=model)
    reviewer = Reviewer(client=client, model=model)

    output_dir = config.OUTPUT_DIR

    print("=" * 60)
    print(f"  Survey Agent — {state.topic}")
    print(f"  Model: {model}")
    print(f"  Focus: {', '.join(state.focus_areas)}")
    print("=" * 60)
    print()

    # ── Phase 1: Planning ────────────────────────────────────────────
    state.current_phase = Phase.PLANNING
    state = lead.run(state)
    save_state_snapshot(state, "01_after_planning.json", output_dir)

    # ── Phase 2: Literature Search ───────────────────────────────────
    state.current_phase = Phase.SEARCHING
    state = searcher.run(state)
    save_state_snapshot(state, "02_after_search.json", output_dir)

    # ── Phase 2b: Search Quality Gate ────────────────────────────────
    additional_queries = lead.review_search_coverage(state)
    if additional_queries:
        state = searcher.run(state, queries=additional_queries)
        save_state_snapshot(state, "02b_after_search_round2.json", output_dir)

    # ── Phase 3: Analysis ────────────────────────────────────────────
    state.current_phase = Phase.ANALYZING
    state = analyst.run(state)
    save_state_snapshot(state, "03_after_analysis.json", output_dir)

    # ── Phase 3b: Outline Refinement ─────────────────────────────────
    state = lead.refine_outline(state)
    save_state_snapshot(state, "03b_after_outline_refinement.json", output_dir)

    # ── Phase 4: Writing ─────────────────────────────────────────────
    state.current_phase = Phase.WRITING
    state = writer.run(state)
    save_state_snapshot(state, "04_after_writing.json", output_dir)

    # ── Phase 5: Review ──────────────────────────────────────────────
    state.current_phase = Phase.REVIEWING
    state = reviewer.run(state)
    save_state_snapshot(state, "05_after_review.json", output_dir)

    # ── Phase 5b: Revision ───────────────────────────────────────────
    sections_to_revise = lead.triage_reviews(state)
    if sections_to_revise:
        state = writer.revise(state, sections_to_revise)
        save_state_snapshot(state, "05b_after_revision.json", output_dir)

    # ── Phase 6: Output ──────────────────────────────────────────────
    state.current_phase = Phase.DONE
    paper_path = save_markdown_paper(state, output_dir=output_dir)
    save_state_snapshot(state, "final_state.json", output_dir)

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Survey paper generated: {paper_path}")
    print(f"  Papers referenced: {state.paper_count()}")
    print(f"  Themes identified: {len(state.themes)}")
    print(f"  Sections written:  {len(state.sections)}")
    review_count = len(state.review_comments)
    revised_count = sum(1 for s in state.sections if s.status == "revised")
    print(f"  Review comments:   {review_count}")
    print(f"  Sections revised:  {revised_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
