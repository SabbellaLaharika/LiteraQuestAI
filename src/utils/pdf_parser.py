import io
from typing import Optional
import pypdf
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.logger import logger
from src.models.state import Paper


class PDFParser:
    """Downloads and extracts introductory/abstract text from research paper PDFs with graceful fallbacks."""

    def __init__(self, timeout_seconds: int = 15, max_pages: int = 3):
        self.timeout_seconds = timeout_seconds
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LiteraQuestAI/1.0 Academic Crawler"
        })

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1.5, min=2, max=5),
        retry=retry_if_exception_type((requests.RequestException, TimeoutError)),
        reraise=True,
    )
    def _download_pdf_bytes(self, pdf_url: str) -> bytes:
        response = self.session.get(self.pdf_url_normalize(pdf_url), timeout=self.timeout_seconds)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            raise ValueError(f"Content-Type is not PDF ({content_type})")
        return response.content

    def pdf_url_normalize(self, url: str) -> str:
        """Ensures arXiv and other URLs point directly to the PDF stream."""
        if "arxiv.org/abs/" in url:
            return url.replace("arxiv.org/abs/", "arxiv.org/pdf/") + ".pdf"
        return url

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extracts text from the first few pages of a PDF (where Abstract and Intro live)."""
        text_chunks: list[str] = []
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            pages_to_read = min(len(reader.pages), self.max_pages)
            for i in range(pages_to_read):
                page_text = reader.pages[i].extract_text()
                if page_text:
                    text_chunks.append(page_text.strip())
        except Exception as e:
            logger.warning(f"PDF Parser: Could not parse PDF bytes: {e}")

        return "\n\n".join(text_chunks)

    def resolve_paper_content(self, paper: Paper) -> str:
        """
        Attempts to download and extract text from the paper's PDF.
        Falls back to the paper's API abstract if the PDF is paywalled or inaccessible.
        """
        if paper.pdf_url:
            try:
                logger.info(f"PDF Parser: Attempting PDF download for '{paper.title[:50]}...' from {paper.pdf_url}")
                pdf_bytes = self._download_pdf_bytes(paper.pdf_url)
                extracted = self.extract_text_from_pdf(pdf_bytes)
                if extracted and len(extracted) > 150:
                    paper.pdf_text_extracted = extracted
                    logger.info(f"PDF Parser: Extracted {len(extracted)} chars of text from PDF for '{paper.title[:50]}'")
                    return extracted
                else:
                    logger.warning(f"PDF Parser: Extracted text too short from PDF for '{paper.title[:50]}'. Using abstract fallback.")
            except Exception as e:
                logger.warning(
                    f"PDF Parser: PDF download/parsing failed or paywalled for '{paper.title[:50]}' ({e}). "
                    "Falling back to metadata abstract."
                )

        # Fallback to abstract
        if paper.abstract:
            logger.info(f"PDF Parser: Relying on API abstract for '{paper.title[:50]}'")
            return paper.abstract

        logger.warning(f"PDF Parser: Neither PDF nor abstract available for '{paper.title[:50]}'")
        return paper.title


pdf_parser = PDFParser()
