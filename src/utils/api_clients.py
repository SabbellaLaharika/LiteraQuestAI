from typing import List, Optional
import arxiv
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from src.config import settings
from src.logger import logger
from src.models.state import Paper


class ArxivClient:
    """Resilient client for querying the arXiv academic database using the arxiv Python package."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3.0,
            num_retries=3,
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _search_arxiv_with_retry(self, query: str, max_results: int) -> List[arxiv.Result]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        return list(self.client.results(search))

    def search(self, query: str, limit: Optional[int] = None) -> List[Paper]:
        max_results = limit or self.max_results
        logger.info(f"Search Agent: Querying arXiv database for query: '{query}' (limit={max_results})")
        papers: List[Paper] = []

        try:
            results = self._search_arxiv_with_retry(query, max_results)
            for res in results:
                paper_id = res.get_short_id()
                year = res.published.year if res.published else None
                authors = [a.name for a in res.authors] if res.authors else ["Unknown"]

                paper = Paper(
                    title=res.title.replace("\n", " ").strip(),
                    authors=authors,
                    abstract=res.summary.replace("\n", " ").strip(),
                    year=year,
                    source="arxiv",
                    source_id=f"arxiv:{paper_id}",
                    url=res.entry_id,
                    pdf_url=res.pdf_url,
                )
                papers.append(paper)

            logger.info(f"Search Agent: Successfully retrieved {len(papers)} papers from arXiv")
        except Exception as e:
            logger.error(f"Search Agent: Exhausted retries querying arXiv for '{query}': {e}")

        return papers


class CrossrefClient:
    """Resilient client for querying the Crossref REST API."""

    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, mailto: Optional[str] = None):
        self.mailto = mailto or settings.crossref_mailto
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"LiteraQuestAI/1.0 (mailto:{self.mailto})",
            "Accept": "application/json",
        })

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _fetch_crossref_with_retry(self, params: dict) -> dict:
        response = self.session.get(self.BASE_URL, params=params, timeout=12)
        response.raise_for_status()
        return response.json()

    def search(self, query: str, limit: int = 5) -> List[Paper]:
        logger.info(f"Search Agent: Querying Crossref REST API for query: '{query}' (limit={limit})")
        papers: List[Paper] = []
        params = {
            "query": query,
            "rows": limit,
            "sort": "relevance",
            "select": "DOI,title,author,abstract,published-print,published-online,URL,link",
        }

        try:
            data = self._fetch_crossref_with_retry(params)
            items = data.get("message", {}).get("items", [])

            for item in items:
                doi = item.get("DOI")
                if not doi:
                    continue

                titles = item.get("title", [])
                title = titles[0] if titles else "Untitled Paper"

                raw_authors = item.get("author", [])
                authors: List[str] = []
                for a in raw_authors:
                    given = a.get("given", "")
                    family = a.get("family", "")
                    name = f"{given} {family}".strip()
                    if name:
                        authors.append(name)
                if not authors:
                    authors = ["Unknown Author"]

                # Extract year
                published = item.get("published-print") or item.get("published-online") or {}
                date_parts = published.get("date-parts", [[]])
                year = date_parts[0][0] if date_parts and date_parts[0] else None

                # Extract abstract (strip JATS tags if present)
                abstract_raw = item.get("abstract", "")
                abstract = (
                    abstract_raw.replace("<jats:p>", "")
                    .replace("</jats:p>", "")
                    .replace("<p>", "")
                    .replace("</p>", "")
                    .strip()
                )

                # Look for direct PDF link in Crossref metadata
                pdf_url: Optional[str] = None
                links = item.get("link", [])
                for link in links:
                    if link.get("content-type") == "application/pdf":
                        pdf_url = link.get("URL")
                        break

                url = item.get("URL") or f"https://doi.org/{doi}"

                paper = Paper(
                    title=title.replace("\n", " ").strip(),
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    source="crossref",
                    source_id=f"doi:{doi}",
                    url=url,
                    pdf_url=pdf_url,
                )
                papers.append(paper)

            logger.info(f"Search Agent: Successfully retrieved {len(papers)} papers from Crossref")
        except Exception as e:
            logger.error(f"Search Agent: Exhausted retries querying Crossref for '{query}': {e}")

        return papers


class SemanticScholarClient:
    """Optional client for Semantic Scholar Graph API with graceful fallback."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.semantic_scholar_api_key
        self.session = requests.Session()
        if self.api_key and self.api_key != "your_api_key_here":
            self.session.headers["x-api-key"] = self.api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2.0, min=2, max=8),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _fetch_semanticscholar_with_retry(self, params: dict) -> dict:
        response = self.session.get(self.BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def search(self, query: str, limit: int = 5) -> List[Paper]:
        logger.info(f"Search Agent: Querying Semantic Scholar API for query: '{query}'")
        papers: List[Paper] = []
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,year,url,openAccessPdf,externalIds",
        }

        try:
            data = self._fetch_semanticscholar_with_retry(params)
            for item in data.get("data", []):
                external_ids = item.get("externalIds") or {}
                source_id = (
                    f"doi:{external_ids['DOI']}"
                    if "DOI" in external_ids
                    else f"corpusId:{item.get('paperId', 'unknown')}"
                )

                authors = [a.get("name", "Unknown") for a in item.get("authors", [])] or ["Unknown"]
                oa_pdf = item.get("openAccessPdf") or {}

                paper = Paper(
                    title=item.get("title", "Untitled").strip(),
                    authors=authors,
                    abstract=item.get("abstract") or "",
                    year=item.get("year"),
                    source="semantic_scholar",
                    source_id=source_id,
                    url=item.get("url"),
                    pdf_url=oa_pdf.get("url"),
                )
                papers.append(paper)
            logger.info(f"Search Agent: Successfully retrieved {len(papers)} papers from Semantic Scholar")
        except Exception as e:
            logger.error(f"Search Agent: Exhausted retries querying Semantic Scholar for '{query}': {e}")

        return papers
