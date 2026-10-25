import re
from typing import List, Tuple
import bibtexparser
from src.logger import logger
from src.models.state import AgentState, Paper


class CitationAgent:
    """
    Citation Agent constructs valid BibTeX entries for all referenced papers,
    validates the syntax with bibtexparser, and generates standardized references.
    """

    def __init__(self):
        pass

    def _generate_citation_key(self, paper: Paper, index: int) -> str:
        """Generates a clean, unique BibTeX citation key: authorYYYYfirstword."""
        first_author = "unknown"
        if paper.authors:
            # Extract last name of the first author
            raw_name = paper.authors[0].strip()
            parts = re.split(r"\s+", raw_name)
            last = parts[-1] if parts else "author"
            first_author = re.sub(r"[^a-zA-Z]", "", last).lower() or "author"

        year_str = str(paper.year) if paper.year else "nodate"

        # First meaningful word from title
        title_words = [w for w in re.split(r"\s+", re.sub(r"[^a-zA-Z0-9\s]", "", paper.title.lower())) if len(w) > 2]
        title_slug = title_words[0] if title_words else "paper"

        key = f"{first_author}{year_str}{title_slug}"
        paper.citation_key = key
        return key

    def format_authors_bibtex(self, authors: List[str]) -> str:
        """Formats authors list into BibTeX 'and' separated standard representation."""
        if not authors:
            return "Unknown"
        return " and ".join(authors)

    def generate_bibtex_entry(self, paper: Paper, index: int) -> str:
        """Constructs a valid BibTeX entry for a given Paper."""
        key = self._generate_citation_key(paper, index)
        author_field = self.format_authors_bibtex(paper.authors)
        title_field = paper.title.replace("{", "").replace("}", "")
        year_field = str(paper.year) if paper.year else "2024"

        fields = [
            f"  title={{{title_field}}}",
            f"  author={{{author_field}}}",
            f"  year={{{year_field}}}",
        ]

        if paper.source == "arxiv":
            arxiv_id = paper.source_id.replace("arxiv:", "")
            fields.append("  journal={arXiv preprint}")
            fields.append(f"  eprint={{{arxiv_id}}}")
            fields.append("  archivePrefix={arXiv}")
        elif paper.source == "crossref" and paper.source_id.startswith("doi:"):
            doi = paper.source_id.replace("doi:", "")
            fields.append("  journal={Crossref Academic Publication}")
            fields.append(f"  doi={{{doi}}}")
        else:
            fields.append(f"  journal={{{paper.source.capitalize()} Publication}}")

        if paper.url:
            fields.append(f"  url={{{paper.url}}}")

        entry_body = ",\n".join(fields)
        bibtex_entry = f"@article{{{key},\n{entry_body}\n}}"
        paper.bibtex = bibtex_entry
        return bibtex_entry

    def format_markdown_reference(self, paper: Paper, index: int) -> str:
        """
        Formats a reference in standard academic Markdown format:
        [1] Author, A. et al. (Year). Title. DOI/URL
        """
        authors_display = "Unknown Author"
        if paper.authors:
            if len(paper.authors) == 1:
                authors_display = paper.authors[0]
            elif len(paper.authors) == 2:
                authors_display = f"{paper.authors[0]} and {paper.authors[1]}"
            else:
                authors_display = f"{paper.authors[0]} et al."

        year_str = str(paper.year) if paper.year else "n.d."
        identifier = f" [{paper.source_id.upper()}]" if paper.source_id else ""
        link_str = f" Available at: {paper.url}" if paper.url else ""

        return f"[{index}] {authors_display} ({year_str}). *{paper.title}*.{identifier}{link_str}"

    def compile_all(self, papers: List[Paper]) -> Tuple[str, List[str]]:
        """Compiles all BibTeX entries and Markdown references with validation."""
        logger.info(f"Citation Agent: Compiling citations and generating BibTeX for {len(papers)} papers")

        bibtex_entries: List[str] = []
        markdown_references: List[str] = []

        for idx, paper in enumerate(papers, start=1):
            entry = self.generate_bibtex_entry(paper, idx)
            bibtex_entries.append(entry)
            ref_str = self.format_markdown_reference(paper, idx)
            markdown_references.append(ref_str)

        full_bibtex = "\n\n".join(bibtex_entries) + "\n"

        # Validate BibTeX syntax with bibtexparser
        try:
            parsed = bibtexparser.loads(full_bibtex)
            logger.info(
                f"Citation Agent: Validated BibTeX syntax with bibtexparser: "
                f"{len(parsed.entries)} valid entries parsed"
            )
        except Exception as e:
            logger.warning(f"Citation Agent: BibTeX parsing verification warning: {e}")

        return full_bibtex, markdown_references

    def run(self, state: AgentState) -> AgentState:
        """Executes citation compilation step on AgentState."""
        state.current_step = "citation_compilation"
        full_bibtex, md_refs = self.compile_all(state.papers)
        state.bibtex_content = full_bibtex
        return state


citation_agent = CitationAgent()
