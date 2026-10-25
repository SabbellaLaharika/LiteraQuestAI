import re
from typing import List
from src.logger import logger
from src.models.state import AgentState, ResearchPlan


class SupervisorAgent:
    """
    Supervisor Agent orchestrates the literature review planning.
    It analyzes the user's research topic, expands it into focused academic subtopics
    and targeted keyword queries for multi-source search agents.
    """

    def __init__(self):
        pass

    def create_plan(self, topic: str) -> ResearchPlan:
        """Analyzes the topic and derives specialized subtopics and search queries."""
        logger.info(f"Supervisor: Planning workflow for topic '{topic}'")

        cleaned_topic = topic.strip()

        # Clean topic tokens to generate sub-queries
        tokens = [w for w in re.split(r"\s+", cleaned_topic) if len(w) > 2]
        base_query = " ".join(tokens)

        # Generate targeted subtopics and query variants for diverse retrieval
        subtopics = [
            f"Foundations and core architectures of {cleaned_topic}",
            f"Recent methodologies and state-of-the-art advances in {cleaned_topic}",
            f"Applications, evaluations, and future challenges in {cleaned_topic}",
        ]

        search_queries = [
            base_query,
            f"{base_query} survey",
            f"{base_query} architectures and methods",
        ]

        plan = ResearchPlan(
            topic=cleaned_topic,
            subtopics=subtopics,
            search_queries=search_queries,
            notes=f"Orchestrated research plan with {len(search_queries)} search vectors across multiple academic databases.",
        )

        logger.info(
            f"Supervisor: Workflow plan established with {len(plan.subtopics)} subtopics "
            f"and {len(plan.search_queries)} search queries"
        )
        return plan

    def run(self, state: AgentState) -> AgentState:
        """Executes supervisor planning step on AgentState."""
        state.current_step = "supervisor_planning"
        state.status = "in_progress"
        state.plan = self.create_plan(state.topic)
        return state


supervisor_agent = SupervisorAgent()
