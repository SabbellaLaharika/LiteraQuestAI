import os
import uuid
from pathlib import Path
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from src.config import LITERATURE_REVIEW_FILE, REFERENCES_BIB_FILE
from src.logger import logger
from src.models.state import AgentState, Paper
from src.state.redis_manager import redis_manager
from src.agents.supervisor import supervisor_agent
from src.agents.search_agent import search_agent
from src.agents.summarizer_agent import summarization_agent
from src.agents.citation_agent import citation_agent


# LangGraph node functions
def supervisor_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Supervisor node: analyzes topic and generates research plan."""
    state = AgentState.model_validate(state_dict)

    # Sync from Redis if available
    cached_state = redis_manager.load_state(state.session_id)
    if cached_state:
        state = cached_state

    logger.info(f"LangGraph: Entering 'supervisor_node' for session '{state.session_id}'")
    updated_state = supervisor_agent.run(state)

    # Save state to Redis
    redis_manager.save_state(updated_state)
    return updated_state.model_dump()


def search_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Web Search node: queries arXiv and Crossref academic sources."""
    state = AgentState.model_validate(state_dict)

    cached_state = redis_manager.load_state(state.session_id)
    if cached_state:
        state = cached_state

    logger.info(f"LangGraph: Entering 'search_node' for session '{state.session_id}'")
    updated_state = search_agent.run(state)

    # Save state to Redis
    redis_manager.save_state(updated_state)
    return updated_state.model_dump()


def summarizer_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Summarization node: parses PDFs and summarizes using local Ollama LLM."""
    state = AgentState.model_validate(state_dict)

    cached_state = redis_manager.load_state(state.session_id)
    if cached_state:
        state = cached_state

    logger.info(f"LangGraph: Entering 'summarizer_node' for session '{state.session_id}'")
    updated_state = summarization_agent.run(state)

    # Save state to Redis
    redis_manager.save_state(updated_state)
    return updated_state.model_dump()


def citation_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Citation node: formats BibTeX entries and references."""
    state = AgentState.model_validate(state_dict)

    cached_state = redis_manager.load_state(state.session_id)
    if cached_state:
        state = cached_state

    logger.info(f"LangGraph: Entering 'citation_node' for session '{state.session_id}'")
    updated_state = citation_agent.run(state)

    # Save state to Redis
    redis_manager.save_state(updated_state)
    return updated_state.model_dump()


def report_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Report generation node: compiles structured literature_review.md
    and references.bib according to contract requirements.
    """
    state = AgentState.model_validate(state_dict)

    cached_state = redis_manager.load_state(state.session_id)
    if cached_state:
        state = cached_state

    state.current_step = "report_generation"
    logger.info(f"LangGraph: Entering 'report_node' - Generating final literature review and BibTeX")

    # Build Markdown Report according to exact Contract Specification:
    # # Literature Review: [Topic]
    # ## Introduction
    # ## Related Work (with ### [Paper Title] and summary block for each paper)
    # ## Conclusion
    # ## References
    topic = state.topic
    intro = state.introduction or "This review provides an analysis of recent scholarly literature."
    conclusion = state.conclusion or "The surveyed literature indicates notable advancements in the field."

    # Build Related Work section
    related_work_blocks = []
    for paper in state.papers:
        authors_str = ", ".join(paper.authors) if paper.authors else "Unknown Authors"
        year_str = f" ({paper.year})" if paper.year else ""
        source_badge = f"`[{paper.source.upper()}]` {paper.source_id}"

        summary_text = paper.summary or "Summary not available for this publication."

        paper_block = (
            f"### {paper.title}\n\n"
            f"- **Authors:** {authors_str}{year_str}\n"
            f"- **Database Identifier:** {source_badge}\n"
            f"- **URL:** [{paper.url}]({paper.url})\n\n"
            f"**Summary of Contributions:**\n\n"
            f"{summary_text}\n"
        )
        related_work_blocks.append(paper_block)

    related_work_content = "\n\n".join(related_work_blocks)

    # Build References section
    reference_items = []
    for idx, paper in enumerate(state.papers, start=1):
        ref_line = citation_agent.format_markdown_reference(paper, idx)
        reference_items.append(ref_line)

    references_content = "\n\n".join(reference_items)

    markdown_report = (
        f"# Literature Review: {topic}\n\n"
        f"## Introduction\n\n"
        f"{intro}\n\n"
        f"## Related Work\n\n"
        f"{related_work_content}\n\n"
        f"## Conclusion\n\n"
        f"{conclusion}\n\n"
        f"## References\n\n"
        f"{references_content}\n"
    )

    state.markdown_report = markdown_report
    state.status = "completed"

    # Write output files
    LITERATURE_REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LITERATURE_REVIEW_FILE, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    logger.info(f"Report Generator: Successfully generated literature review at '{LITERATURE_REVIEW_FILE}'")

    if state.bibtex_content:
        with open(REFERENCES_BIB_FILE, "w", encoding="utf-8") as f:
            f.write(state.bibtex_content)
        logger.info(f"Report Generator: Successfully generated BibTeX file at '{REFERENCES_BIB_FILE}'")

    # Final Redis save
    redis_manager.save_state(state)
    return state.model_dump()


def build_workflow() -> Any:
    """Constructs the LangGraph state machine linking all agents sequentially."""
    workflow = StateGraph(dict)

    # Add agent nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("search", search_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("citation", citation_node)
    workflow.add_node("report", report_node)

    # Define execution edges
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "search")
    workflow.add_edge("search", "summarizer")
    workflow.add_edge("summarizer", "citation")
    workflow.add_edge("citation", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


# Compiled LangGraph application
research_app = build_workflow()


def run_research_workflow(topic: str, session_id: str | None = None) -> AgentState:
    """Executes the entire multi-agent workflow for a given topic."""
    sess_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
    initial_state = AgentState(
        session_id=sess_id,
        topic=topic,
        status="pending",
        current_step="initialized",
    )

    # Save initial state in Redis
    redis_manager.save_state(initial_state)

    logger.info(f"Orchestrator: Starting LangGraph workflow for session '{sess_id}' on topic '{topic}'")

    final_state_dict = research_app.invoke(initial_state.model_dump())
    final_state = AgentState.model_validate(final_state_dict)

    logger.info(f"Orchestrator: Workflow completed successfully for session '{sess_id}' with status '{final_state.status}'")
    return final_state
