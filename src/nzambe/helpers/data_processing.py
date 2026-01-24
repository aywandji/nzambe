import json
import logging
import os
import re
from pathlib import Path
from typing import Sequence

import ollama
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
)
from llama_index.core.constants import DEFAULT_CHUNK_SIZE
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser.text.sentence import (
    SENTENCE_CHUNK_OVERLAP,
    SentenceSplitter,
)
from llama_index.core.schema import BaseNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from nzambe.config import nzambe_settings
from nzambe.constants import (
    VERSE_PATTERN,
    OLD_TESTAMENT_FIRST_BOOK,
    OLD_TESTAMENT_LAST_BOOK,
    NEW_TESTAMENT_FIRST_BOOK,
    NEW_TESTAMENT_LAST_BOOK,
    ALTERNATIVE_BOOK_NAMES,
    GUTENBERG_END_MARKER,
)

logger = logging.getLogger(__name__)


def extract_testament_books_names(
    bible_text: str, first_book_name: str, last_book_name: str
) -> list[str]:
    """
    Extracts the names of books within from a given bible text. The books names are assumed to be all listed and
     separated by a newline character.

    :param bible_text: The full text of the Bible.
    :param first_book_name: The name of the first book in the testament.
    :param last_book_name: The name of the last book in the testament.
    :return: A list of book names within the specified testament.
    """
    testament_books = (
        bible_text.split(first_book_name + "\n", 1)[1]
        .split(last_book_name, 1)[0]
        .split("\n")
    )
    testament_books.append(last_book_name)
    testament_books = [
        book for book in [first_book_name] + testament_books if len(book) > 0
    ]
    return testament_books


def extract_book_text(bible_text: str, book_name: str, next_book_name: str):
    return (
        bible_text.split(f"{book_name}\n\n\n")[1]
        .split(next_book_name)[0]
        .strip()
        .replace("***", "")
        .replace("The New Testament of the King James Bible", "")
        .strip()
    )


def extract_book_verses(book_text: str):
    # Pattern: one or more word characters, followed by a colon,
    # followed by one or more word characters.
    verses = re.split(VERSE_PATTERN, book_text)[1:]

    verses_dict = {}
    for i in range(0, len(verses), 2):
        verse_number = verses[i]
        verse_text = verses[i + 1]
        verses_dict[verse_number] = verse_text.replace("\n", " ").strip()
    return verses_dict


def split_bible_text_by_books(
    bible_text_path: str, destination_directory: str | None = None
):
    """
    Text coming from https://www.gutenberg.org/cache/epub/10/pg10.txt
    :param bible_text_path:
    :return:
    """
    with open(bible_text_path, "r") as f:
        bible_text = f.read()

    # old testament books
    old_testament_books = extract_testament_books_names(
        bible_text, OLD_TESTAMENT_FIRST_BOOK, OLD_TESTAMENT_LAST_BOOK
    )

    # new testament books
    new_testament_books = extract_testament_books_names(
        bible_text, NEW_TESTAMENT_FIRST_BOOK, NEW_TESTAMENT_LAST_BOOK
    )

    bible_dict = {}
    all_books = old_testament_books + new_testament_books
    all_books.append(GUTENBERG_END_MARKER)
    for i in range(len(all_books) - 1):
        book_name = all_books[i]
        alternative_book_name = ALTERNATIVE_BOOK_NAMES.get(book_name, book_name)

        book_text = extract_book_text(
            bible_text, alternative_book_name, all_books[i + 1]
        )
        bible_dict[book_name] = extract_book_verses(book_text)
        if destination_directory:
            Path(destination_directory).mkdir(parents=True, exist_ok=True)
            book_path = f"{destination_directory}/{book_name}.txt"
            if not os.path.exists(book_path):
                logger.info(f"saving book {book_name} to {destination_directory}")
                # save book text as a separate document
                with open(f"{destination_directory}/{book_name}.txt", "w") as f:
                    f.write(book_text)

    return bible_dict


async def from_documents_to_nodes(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
    paragraph_separator: str = "\n\n",
    num_workers: int | None = None,
) -> Sequence[BaseNode]:
    """
    Transforms a list of documents into nodes suitable for further processing. This function
    uses a configurable pipeline of transformations and optionally supports multiprocessing
    for performance optimization.

    :param documents: A list of `Document` instances to be processed.
    :param chunk_size: Integer defining the maximum size of text chunks to create during
        sentence splitting.
    :param chunk_overlap: Integer defining overlapping sentences between consecutive text
        chunks to maintain context.
    :param paragraph_separator: String used to delimit paragraphs in the document
        during processing.
    :param num_workers: Optional integer specifying the number of worker threads to use
        for parallel processing. If None, the function runs transformations with embeddings
        sequentially and asynchronously.

    :return: A list of processed nodes after applying transformations to the documents.
    """
    # create the pipeline with transformations
    transformations = [
        SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator=paragraph_separator,
        )
    ]
    if num_workers is None:
        # no multiprocessing for transformations, so let's run embedding asynchronously
        transformations.append(Settings.embed_model)

    pipeline = IngestionPipeline(transformations=transformations)

    if num_workers is None:
        nodes = await pipeline.arun(documents=documents)
    else:
        nodes = pipeline.run(documents=documents, num_workers=num_workers)

    return nodes


def get_document_split_chunk_size() -> int:
    if isinstance(Settings.embed_model, OllamaEmbedding):
        embed_model_context_length = None
        show_response = ollama.show(Settings.embed_model.model_name)
        if show_response.modelinfo is not None:
            for k, v in show_response.modelinfo.items():
                if "context_length" in k:
                    embed_model_context_length = int(v)
                    break
        if embed_model_context_length is None:
            raise Exception("Could not determine embedding model context length.")
    elif isinstance(Settings.embed_model, OpenAIEmbedding):
        embed_model_context_length = 8192  # default defined at
        # https://platform.openai.com/docs/guides/embeddings/#embedding-models
    else:
        raise Exception(f"Unsupported embedding model: {Settings.embed_model}")

    # set a chunk size smaller than embedding model context length and llm context window
    # to reserve space for the user query and the qa prompt.
    # the smaller the value, the more precise the embeddings will be.
    embed_model_input_length = int(0.5 * embed_model_context_length)
    llm_engine_context_length = int(
        0.5 * nzambe_settings.llm.query_model.context_window
    )
    return min(embed_model_input_length, llm_engine_context_length)


# TODO: turn this into a class that will have a load_or_build method which will return an index instance
async def build_documents_index(
    index_storage_path: Path,
    document_split_chunk_overlap: int,
    input_data_files: list[str] | None = None,
    paragraph_separator: str = "\n\n",
    insert_batch_size: int = 2048,
    num_workers: int | None = None,
) -> VectorStoreIndex | BaseIndex:
    if nzambe_settings.env not in ("local", "test"):
        raise Exception("Documents index can only be built locally for now.")
    elif not (
        isinstance(Settings.embed_model, OllamaEmbedding)
        or isinstance(Settings.embed_model, OpenAIEmbedding)
    ):
        raise Exception(
            f"Unsupported embedding model: {Settings.embed_model} for local serving"
        )

    if os.path.exists(index_storage_path) and len(os.listdir(index_storage_path)) > 0:
        logger.info("loading index from disk...")
        storage_context = StorageContext.from_defaults(
            persist_dir=str(index_storage_path)
        )
        index = load_index_from_storage(storage_context)
    else:
        if input_data_files is None:
            raise Exception("input_data_files must be provided when building index.")

        # TODO: remove line below
        input_data_files = input_data_files[:3]

        # saving index creation metadata (useful for loading back the index)
        embedding_model_conf = nzambe_settings.llm.embedding_model.model_dump()
        if embedding_model_conf["name"] != Settings.embed_model.model_name:
            raise Exception(
                f"Embedding model name mismatch: {embedding_model_conf['name']} != {Settings.embed_model.model_name}"
            )

        metadata = {
            "embedding_model_conf": embedding_model_conf,
            "index_conf": nzambe_settings.index.model_dump(),
        }
        index_storage_path.mkdir(parents=True, exist_ok=True)
        with open(os.path.join(index_storage_path, "nzambe_metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info("building index from scratch...")
        # load documents
        documents = SimpleDirectoryReader(
            input_files=input_data_files, exclude_hidden=False
        ).load_data()

        document_split_chunk_size = get_document_split_chunk_size()
        nodes = await from_documents_to_nodes(
            documents,
            chunk_size=document_split_chunk_size,
            chunk_overlap=document_split_chunk_overlap,
            paragraph_separator=paragraph_separator,
            num_workers=num_workers,
        )
        index = VectorStoreIndex(
            nodes=nodes,
            insert_batch_size=insert_batch_size,
            show_progress=True,
            embed_model=Settings.embed_model,
        )
        logger.info("persisting index to disk...")
        index.storage_context.persist(index_storage_path)
    return index
