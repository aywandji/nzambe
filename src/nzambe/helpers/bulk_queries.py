import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from nzambe.helpers.client import query_server

logger = logging.getLogger(__name__)


def process_question(question: str) -> tuple[str, str]:
    """Query server and return (question, answer) tuple."""
    try:
        server_answer, docs = query_server(question)
        return question, server_answer
    except Exception as e:
        logger.error(f"Failed to query '{question}': {e}")
        return question, f"ERROR: {e}"


if __name__ == "__main__":
    data_folder_path = Path(__file__).parents[2] / ".debug/data/bible"

    with open(data_folder_path / "qa_dataset.json", "r") as f:
        questions_dataset = json.load(f)

    questions = list(questions_dataset["queries"].values())

    # Adjust max_workers to simulate different numbers of concurrent clients
    # E.g., 10 = 10 concurrent clients, 50 = 50 concurrent clients
    max_workers = 50

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_answers = [executor.submit(process_question, q) for q in questions]

        # Process results as they complete
        for future in tqdm(as_completed(future_answers), total=len(questions)):
            question, answer = future.result()
            logger.info(f"Q: {question}")
            logger.info(answer)
