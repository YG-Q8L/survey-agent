"""
Academic paper search tools.

Wraps Semantic Scholar and arXiv APIs. Returns Paper dataclass instances.
Both APIs are free and require no authentication for basic use.
"""

from __future__ import annotations

import time

import arxiv
import requests

from state import Paper

# ── Semantic Scholar ─────────────────────────────────────────────────

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
SS_FIELDS = "paperId,title,year,abstract,citationCount,authors,url"


def search_semantic_scholar(query: str, limit: int = 20) -> list[Paper]:
    """Search Semantic Scholar. Returns papers with abstracts only."""
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
    params = {"query": query, "limit": limit, "fields": SS_FIELDS}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        papers = []
        for item in data:
            if not item.get("abstract"):
                continue
            papers.append(Paper(
                paper_id=item["paperId"],
                title=item["title"],
                authors=[a["name"] for a in item.get("authors", [])],
                year=item.get("year") or 0,
                abstract=item.get("abstract", ""),
                citation_count=item.get("citationCount") or 0,
                url=item.get("url", ""),
                source="semantic_scholar",
            ))
        return papers
    except requests.RequestException as e:
        print(f"[Search] Semantic Scholar error for '{query}': {e}")
        return []


# ── arXiv ────────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 15) -> list[Paper]:
    """Search arXiv. Note: arXiv does not provide citation counts."""
    try:
        client = arxiv.Client(delay_seconds=3)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for result in client.results(search):
            papers.append(Paper(
                paper_id=result.entry_id,
                title=result.title,
                authors=[a.name for a in result.authors],
                year=result.published.year,
                abstract=result.summary,
                citation_count=0,
                url=result.entry_id,
                source="arxiv",
            ))
        return papers
    except Exception as e:
        print(f"[Search] arXiv error for '{query}': {e}")
        return []


# ── Deduplication ────────────────────────────────────────────────────

def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """Remove duplicate papers by title similarity (trigram Jaccard)."""
    seen: list[str] = []
    unique: list[Paper] = []
    for p in papers:
        normalized = p.title.lower().strip()
        is_dup = any(_title_similarity(normalized, s) > 0.85 for s in seen)
        if not is_dup:
            seen.append(normalized)
            unique.append(p)
    return unique


def _title_similarity(a: str, b: str) -> float:
    """Jaccard similarity on character trigrams."""
    if len(a) < 3 or len(b) < 3:
        return 0.0
    tri_a = {a[i : i + 3] for i in range(len(a) - 2)}
    tri_b = {b[i : i + 3] for i in range(len(b) - 2)}
    intersection = len(tri_a & tri_b)
    union = len(tri_a | tri_b)
    return intersection / union if union else 0.0


# ── Batch Search ─────────────────────────────────────────────────────

def search_all(
    queries: list[str],
    papers_per_query: int = 20,
) -> list[Paper]:
    """
    Run all queries against both APIs, deduplicate, and return results.

    Adds a small delay between Semantic Scholar calls to be polite.
    """
    all_papers: list[Paper] = []
    for i, query in enumerate(queries):
        print(f"[Search] ({i + 1}/{len(queries)}) Searching: {query}")
        ss_results = search_semantic_scholar(query, limit=papers_per_query)
        ax_results = search_arxiv(query, max_results=papers_per_query)
        all_papers.extend(ss_results)
        all_papers.extend(ax_results)
        # polite delay for Semantic Scholar shared rate pool
        if i < len(queries) - 1:
            time.sleep(1)
    deduped = deduplicate_papers(all_papers)
    print(f"[Search] Total: {len(all_papers)} raw → {len(deduped)} after dedup")
    return deduped
