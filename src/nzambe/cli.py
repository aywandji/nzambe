"""
Command-line interface for Nzambe RAG system.

Usage:
    nzambe health [--base-url URL]
    nzambe query --question "..." [--base-url URL]
    nzambe retrieve --question "..." [--base-url URL]
    nzambe generate-questions [OPTIONS]
    nzambe server [--host HOST] [--port PORT] [--reload] [--workers N]
    nzambe eval [--last-n-hours N] [--num-traces-limit N]
"""

import logging

import click

from nzambe.helpers.client import health_check, query_server
from nzambe.constants import NZAMBE_SERVER_DEFAULT_BASE_URL

logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def cli():
    """Nzambe - Question answering system for holy books using RAG."""
    pass


@cli.command()
@click.option(
    "--base-url",
    default=None,
    help=f"Server base URL (default: env NZAMBE_API_URL or {NZAMBE_SERVER_DEFAULT_BASE_URL})",
)
def health(base_url):
    """Check the health status of the Nzambe server."""
    try:
        status = health_check(base_url)
        click.echo(f"Server Status: {status.status}")
        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option(
    "--question",
    "-q",
    required=True,
    help="Question to ask the RAG system",
)
@click.option(
    "--base-url",
    default=None,
    help=f"Server base URL (default: env NZAMBE_API_URL or {NZAMBE_SERVER_DEFAULT_BASE_URL})",
)
@click.option(
    "--docs-only",
    help="Only retrieve relevant documents",
    is_flag=True,
)
def query(question, base_url, docs_only):
    """Query the RAG system with a question. Return the answer and relevant documents. Can retrieve only
    relevant documents using the --docs-only flag."""
    try:
        answer, docs = query_server(question, base_url, docs_only)
        if answer:
            click.echo(f"\nAnswer: {answer}\n")
        if docs:
            click.echo("Relevant Documents:")
            for i, doc in enumerate(docs, 1):
                click.echo(f"   {doc}")
        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option(
    "--ollama-model",
    "-m",
    required=True,
    help="Name of the Ollama model to use for question generation",
)
@click.option(
    "--index-storage-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    help="Directory containing the stored index",
)
@click.option(
    "--num-questions-per-node",
    "-n",
    default=2,
    type=int,
    help="Number of questions to generate per node (default: 2)",
)
@click.option(
    "--num-nodes-to-sample",
    "-s",
    default=20,
    type=int,
    help="Number of nodes to sample for question generation (default: 20)",
)
@click.option(
    "--random-seed",
    "-r",
    default=123,
    type=int,
    help="Random seed for reproducibility (default: 123)",
)
@click.option(
    "--questions-dataset-path",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    help="Path to save/update the questions dataset JSON file",
)
def generate_questions(
    ollama_model,
    index_storage_dir,
    num_questions_per_node,
    num_nodes_to_sample,
    random_seed,
    questions_dataset_path,
):
    """Generate questions from documents in the index for evaluation purposes."""
    from nzambe.helpers.eval import generate_new_questions_from_index

    try:
        click.echo("Generating questions from index...")
        click.echo(f"  Ollama model: {ollama_model}")
        click.echo(f"  Index directory: {index_storage_dir}")
        click.echo(f"  Questions per node: {num_questions_per_node}")
        click.echo(f"  Nodes to sample: {num_nodes_to_sample}")
        click.echo(f"  Random seed: {random_seed}")
        if questions_dataset_path:
            click.echo(f"  Output path: {questions_dataset_path}")
        click.echo()

        _ = generate_new_questions_from_index(
            ollama_model_name=ollama_model,
            index_storage_dir=index_storage_dir,
            num_questions_per_node=num_questions_per_node,
            num_nodes_to_sample=num_nodes_to_sample,
            random_seed=random_seed,
            questions_dataset_path=questions_dataset_path,
        )

        click.echo("\n✓ Successfully generated questions!")
        if questions_dataset_path:
            click.echo(f"  Saved to: {questions_dataset_path}")
        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the server to (default: 0.0.0.0)",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind the server to (default: 8000)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload on code changes (for development)",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    help="Number of worker processes (for production, default: 1)",
)
def server(host, port, reload, workers):
    """Start the Nzambe FastAPI server."""
    try:
        import uvicorn

        click.echo(f"Starting Nzambe server on {host}:{port}...")
        if reload:
            click.echo("Auto-reload enabled (development mode)")
        if workers > 1:
            click.echo(f"Using {workers} worker processes")

        uvicorn.run(
            "nzambe.server.server:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers
            if not reload
            else 1,  # reload doesn't work with multiple workers
        )
    except ImportError:
        click.echo("Error: uvicorn is required to run the server.", err=True)
        click.echo("It should be installed with fastapi[standard].", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option(
    "--last-n-hours",
    default=60,
    type=int,
    help="Fetch traces from the last N hours (default: 60)",
)
@click.option(
    "--num-traces-limit",
    default=1,
    type=int,
    help="Maximum number of traces to evaluate (default: 1)",
)
def eval(last_n_hours, num_traces_limit):
    """Run an evaluation benchmark on recent traces from Langfuse."""
    from nzambe.helpers.eval import run_nightly_benchmark

    try:
        click.echo("Starting evaluation benchmark...")
        click.echo(f"  Time range: Last {last_n_hours} hours")
        click.echo(f"  Trace limit: {num_traces_limit}")
        click.echo()

        run_nightly_benchmark(
            last_n_hours=last_n_hours, num_traces_limit=num_traces_limit
        )

        click.echo("\n✓ Evaluation complete!")
        return 0
    except Exception as e:
        click.echo(f"Error running evaluation: {e}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
