"""
Survey Agent — Multi-agent system for writing technical survey papers.

Usage:
    # Anthropic (default)
    export ANTHROPIC_API_KEY="sk-ant-..."
    python main.py --topic ich

    # OpenRouter
    export OPENROUTER_API_KEY="sk-or-..."
    python main.py --topic ich --provider openrouter --model anthropic/claude-sonnet-4

    # Custom model
    python main.py --topic ich --model claude-opus-4-6
"""

import argparse

import config
from llm_client import create_client
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
    parser.add_argument(
        "--provider", default="anthropic", choices=["anthropic", "openrouter"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (default: from config, e.g. claude-sonnet-4-20250514)",
    )
    args = parser.parse_args()

    config.load_topic(args.topic)

    # ── Initialize ───────────────────────────────────────────────────
    model = args.model or config.MODEL
    client = create_client(args.provider, model)

    state = PaperState(
        topic=config.TOPIC,
        topic_en=config.TOPIC_EN,
        focus_areas=list(config.FOCUS_AREAS),
    )

    lead = ResearchLead(client=client)
    searcher = LiteratureSearcher(client=client)
    analyst = Analyst(client=client)
    writer = Writer(client=client)
    reviewer = Reviewer(client=client)

    output_dir = config.OUTPUT_DIR

    print("=" * 60)
    print(f"  Survey Agent — {state.topic}")
    print(f"  Provider: {args.provider}")
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
