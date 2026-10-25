import re
from typing import List, Set
from src.logger import logger
from src.models.state import AgentState, Paper
from src.utils.api_clients import ArxivClient, CrossrefClient, SemanticScholarClient


class SearchAgent:
    """
    Web Search Agent queries at least two distinct academic sources (arXiv and Crossref / Semantic Scholar),
    retrieves paper metadata, normalizes records, and deduplicates papers.
    """

    def __init__(self, target_per_source: int = 4):
        self.target_per_source = target_per_source
        self.arxiv_client = ArxivClient(max_results=target_per_source)
        self.crossref_client = CrossrefClient()
        self.semantic_scholar_client = SemanticScholarClient()

    def _normalize_title(self, title: str) -> str:
        """Normalizes titles for deduplication comparison."""
        return re.sub(r"[^a-zA-Z0-9]", "", title.lower())

    def search_all_sources(self, queries: List[str]) -> List[Paper]:
        """Retrieves and merges papers from at least two distinct online academic sources."""
        seen_titles: Set[str] = set()
        seen_ids: Set[str] = set()
        aggregated_papers: List[Paper] = []

        primary_query = queries[0] if queries else "artificial intelligence"

        # Source 1: arXiv
        arxiv_papers: List[Paper] = []
        try:
            arxiv_papers = self.arxiv_client.search(primary_query, limit=self.target_per_source)
            logger.info(f"Search Agent: Found {len(arxiv_papers)} papers from arXiv")
        except Exception as e:
            logger.error(f"Search Agent: Error during arXiv search: {e}")

        for paper in arxiv_papers:
            norm_title = self._normalize_title(paper.title)
            if norm_title not in seen_titles and paper.source_id not in seen_ids:
                seen_titles.add(norm_title)
                seen_ids.add(paper.source_id)
                aggregated_papers.append(paper)

        # Source 2: Crossref REST API
        crossref_papers: List[Paper] = []
        try:
            crossref_papers = self.crossref_client.search(primary_query, limit=self.target_per_source)
            logger.info(f"Search Agent: Found {len(crossref_papers)} papers from Crossref")
        except Exception as e:
            logger.error(f"Search Agent: Error during Crossref search: {e}")

        # If Crossref returns empty, attempt Semantic Scholar as resilient secondary
        if not crossref_papers:
            logger.warning("Search Agent: Crossref returned 0 papers; attempting Semantic Scholar fallback...")
            try:
                crossref_papers = self.semantic_scholar_client.search(primary_query, limit=self.target_per_source)
                logger.info(f"Search Agent: Found {len(crossref_papers)} papers from Semantic Scholar")
            except Exception as e:
                logger.error(f"Search Agent: Error during Semantic Scholar search: {e}")

        for paper in crossref_papers:
            norm_title = self._normalize_title(paper.title)
            if norm_title not in seen_titles and paper.source_id not in seen_ids:
                seen_titles.add(norm_title)
                seen_ids.add(paper.source_id)
                aggregated_papers.append(paper)

        sources_present = {p.source for p in aggregated_papers}
        logger.info(
            f"Search Agent: Aggregated {len(aggregated_papers)} unique papers across {len(sources_present)} "
            f"distinct sources: {list(sources_present)}"
        )

        return aggregated_papers

    def run(self, state: AgentState) -> AgentState:
        """Executes search agent step on AgentState."""
        state.current_step = "search_retrieval"
        queries = state.plan.search_queries if state.plan and state.plan.search_queries else [state.topic]
        papers = self.search_all_sources(queries)
        state.papers = papers
        return state


search_agent = SearchAgent()
