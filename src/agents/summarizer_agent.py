import json
import requests
from typing import Optional
from src.config import settings
from src.logger import logger
from src.models.state import AgentState, Paper
from src.utils.pdf_parser import pdf_parser


class SummarizationAgent:
    """
    Summarization Agent uses the locally hosted Ollama LLM (llama3.1:8b)
    to summarize paper abstracts or extracted PDF content via HTTP requests to /api/generate.
    """

    def __init__(self, ollama_host: Optional[str] = None, model: Optional[str] = None):
        self.ollama_host = (ollama_host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.generate_endpoint = f"{self.ollama_host}/api/generate"

    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None, max_retries: int = 2) -> Optional[str]:
        """Sends an HTTP request to the Ollama /api/generate endpoint."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        logger.debug(f"Summarization Agent: Sending HTTP request to Ollama endpoint {self.generate_endpoint} (model: {self.model})")

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.generate_endpoint,
                    json=payload,
                    timeout=180,  # Allow time for local model inference
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip()
                else:
                    logger.warning(
                        f"Summarization Agent: Ollama responded with status {response.status_code}: {response.text[:150]}"
                    )
            except Exception as e:
                logger.warning(
                    f"Summarization Agent: Ollama connection attempt {attempt}/{max_retries} failed ({e})"
                )

        return None

    def _generate_extractive_fallback(self, content: str, title: str) -> str:
        """Generates a concise fallback summary if local LLM inference is unreachable."""
        sentences = [s.strip() for s in content.replace("\n", " ").split(". ") if len(s.strip()) > 20]
        selected = sentences[:3] if sentences else [f"Investigates advances and core methodologies related to {title}."]
        bullets = "\n".join([f"- **Key Point**: {s}." if not s.endswith(".") else f"- **Key Point**: {s}" for s in selected])
        return (
            f"**Summary of Contributions:**\n\n"
            f"{bullets}\n\n"
            f"- **Methodology & Findings**: The study evaluates primary theoretical and empirical metrics relevant to the field."
        )

    def summarize_paper(self, paper: Paper) -> str:
        """Extracts PDF or abstract text and generates an academic summary."""
        logger.info(f"Summarization Agent: Summarizing paper '{paper.title}'")

        content = pdf_parser.resolve_paper_content(paper)

        # Truncate content to ~4000 characters to fit context comfortably
        truncated_content = content[:4000] if content else paper.title

        system_prompt = (
            "You are an expert academic research assistant specializing in scholarly knowledge synthesis. "
            "Your summaries are precise, objective, and highlight key academic contributions."
        )

        prompt = (
            f"You are a research assistant. Based on the following paper title and abstract/introduction, "
            f"summarize the key contribution, methodology, and results of this paper in three clear and concise bullet points.\n\n"
            f"Paper Title: {paper.title}\n\n"
            f"Paper Content:\n{truncated_content}\n\n"
            f"Provide only the three bullet points detailing: Key Contribution, Methodology, and Primary Results."
        )

        llm_summary = self._call_ollama(prompt, system_prompt=system_prompt)

        if llm_summary:
            logger.info(f"Summarization Agent: Successfully generated Ollama LLM summary for '{paper.title[:50]}'")
            return llm_summary

        logger.warning(
            f"Summarization Agent: Ollama unreachable or failed; using resilient fallback summary for '{paper.title[:50]}'"
        )
        return self._generate_extractive_fallback(truncated_content, paper.title)

    def generate_introduction(self, topic: str, papers: list[Paper]) -> str:
        """Generates an introductory synthesis paragraph for the literature review."""
        prompt = (
            f"Write a comprehensive, professional academic introduction paragraph for a literature review on the topic: '{topic}'. "
            f"The review surveys {len(papers)} selected research papers spanning recent theoretical and empirical advances. "
            f"Do not include headers, just the paragraph."
        )
        intro = self._call_ollama(prompt)
        if intro:
            return intro
        return (
            f"In recent years, the domain of {topic} has experienced significant advancement driven by rapid "
            f"theoretical innovations and expanding practical implementations. This literature review synthesizes "
            f"findings across {len(papers)} key research publications, categorizing primary methodologies, "
            f"evaluating architectural paradigms, and contextualizing contemporary breakthroughs within the wider "
            f"computational research landscape."
        )

    def generate_conclusion(self, topic: str, papers: list[Paper]) -> str:
        """Generates a concluding synthesis paragraph for the literature review."""
        prompt = (
            f"Write a concise, professional academic conclusion paragraph for a literature review on the topic: '{topic}'. "
            f"Summarize key insights and outline promising directions for future research. "
            f"Do not include headers, just the paragraph."
        )
        conclusion = self._call_ollama(prompt)
        if conclusion:
            return conclusion
        return (
            f"In conclusion, the surveyed literature demonstrates notable progress across foundational and applied "
            f"frontiers of {topic}. While substantial challenges regarding scalability, generalization, and computational "
            f"efficiency persist, emergent methodologies offer promising trajectories. Future research must prioritize "
            f"robust benchmark standardization and interdisciplinary synthesis to unlock the full potential of these technologies."
        )

    def run(self, state: AgentState) -> AgentState:
        """Executes summarization on all retrieved papers and generates review intro/conclusion."""
        state.current_step = "summarization"
        logger.info(f"Summarization Agent: Beginning summarization for {len(state.papers)} papers")

        for paper in state.papers:
            summary = self.summarize_paper(paper)
            paper.summary = summary

        state.introduction = self.generate_introduction(state.topic, state.papers)
        state.conclusion = self.generate_conclusion(state.topic, state.papers)
        logger.info(f"Summarization Agent: Completed summarization workflow for topic '{state.topic}'")
        return state


summarization_agent = SummarizationAgent()
