import argparse
import sys
from src.config import settings, LITERATURE_REVIEW_FILE, REFERENCES_BIB_FILE, LOG_FILE
from src.logger import logger
from src.graph.workflow import run_research_workflow


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="LiteraQuestAI: Multi-Agent Research Assistant using LangGraph, Redis, and Academic APIs."
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Academic research topic to generate a literature review for.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the interactive Web Dashboard server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.web_port,
        help="Port for the interactive Web Dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_arguments()

    if args.serve:
        try:
            import uvicorn
            from src.web.server import app
            logger.info(f"Starting Web Dashboard server at http://{settings.web_host}:{args.port}")
            uvicorn.run(app, host=settings.web_host, port=args.port, log_level="info")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            sys.exit(1)

    if not args.topic:
        logger.error("Error: --topic argument is required when not in --serve mode.")
        print("Usage: python main.py --topic \"Your Research Topic Here\"")
        print("   or: python main.py --serve")
        sys.exit(1)

    topic = args.topic.strip()
    logger.info(f"CLI: Received research topic request: '{topic}'")

    try:
        final_state = run_research_workflow(topic=topic)

        # Confirm output files existence
        if LITERATURE_REVIEW_FILE.exists() and REFERENCES_BIB_FILE.exists():
            logger.info("=" * 60)
            logger.info("RESEARCH WORKFLOW COMPLETED SUCCESSFULLY!")
            logger.info(f"Generated Literature Review: {LITERATURE_REVIEW_FILE}")
            logger.info(f"Generated BibTeX Citations:  {REFERENCES_BIB_FILE}")
            logger.info(f"Execution Log:               {LOG_FILE}")
            logger.info("=" * 60)
            sys.exit(0)
        else:
            logger.warning("Workflow finished but one or more output files are missing.")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Workflow execution cancelled by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error during research workflow execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
