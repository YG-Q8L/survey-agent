# Survey Agent

Multi-agent system for writing technical survey papers, powered by Claude.

## Architecture

```
              ┌──────────────┐
              │ Research Lead │  Orchestrator: planning, quality gates
              └──────┬───────┘
        ┌────────────┼────────────┬────────────┐
        ▼            ▼            ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Literature│ │ Analyst  │ │  Writer  │ │ Reviewer │
  │ Searcher │ │          │ │          │ │          │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**5 Agents, 1 Shared State, 6-Phase Pipeline:**

1. **Planning** — Research Lead generates search queries and paper outline
2. **Searching** — Literature Searcher queries Semantic Scholar + arXiv
3. **Analyzing** — Analyst clusters papers into themes, identifies gaps
4. **Writing** — Writer drafts each section based on themes and outline
5. **Reviewing** — Reviewer provides structured feedback per section
6. **Revision** — Writer revises sections flagged by the Reviewer

Agents communicate through a shared `PaperState` object — no message queues, no framework overhead.

## Quick Start

```bash
# Clone
git clone https://github.com/YG-Q8L/survey-agent.git
cd survey-agent

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run with a topic
python main.py --topic ich       # 非遗数字化保护
python main.py --topic example   # template for your own topic
```

## Adding a New Topic

Copy `topics/example.py` and fill in your own:

```python
# topics/my_topic.py
TOPIC = "Your Research Topic"
TOPIC_EN = "Your Research Topic (English)"
FOCUS_AREAS = [
    "sub-topic 1",
    "sub-topic 2",
    "sub-topic 3",
]
```

Then run: `python main.py --topic my_topic`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `SURVEY_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `SURVEY_PAPERS_PER_QUERY` | `20` | Max papers per search query |
| `SURVEY_OUTPUT_DIR` | `output` | Output directory |

## Project Structure

```
survey-agent/
├── main.py              # Entry point — python main.py --topic ich
├── config.py            # Model, search, and LLM parameter settings
├── state.py             # PaperState shared across all agents
├── topics/              # One file per survey topic
│   ├── ich.py           # 非遗数字化保护
│   └── example.py       # Template for new topics
├── agents/
│   ├── base.py          # BaseAgent: LLM calls, retry, JSON parsing
│   ├── research_lead.py # Orchestrator — plans, reviews, triages
│   ├── literature_searcher.py  # Searches academic APIs
│   ├── analyst.py       # Clusters papers into themes
│   ├── writer.py        # Drafts and revises sections
│   └── reviewer.py      # Structured peer review
├── tools/
│   ├── search.py        # Semantic Scholar + arXiv API wrappers
│   └── file_io.py       # State snapshots + Markdown assembly
├── prompts/             # Editable system prompts per agent
└── output/              # Generated at runtime
```

## Output

After running, check the `output/` directory:

- `survey_paper.md` — The final assembled paper
- `01_after_planning.json` ... `final_state.json` — State snapshots at each phase (useful for debugging)

## Customization

**Switch topic**: Add a new file in `topics/` and run with `--topic <name>`.

**Domain-specific tuning**: Edit prompt templates in `prompts/*.txt` to add domain knowledge or adjust writing style.

**Model selection**: Default is Sonnet for cost efficiency. Set `SURVEY_MODEL=claude-opus-4-6` for maximum quality.

## Dependencies

Only 3 packages — no frameworks, no vector databases:

- `anthropic` — Claude API
- `requests` — Semantic Scholar API
- `arxiv` — arXiv API

## License

MIT
