"""Repaired retrieval pipeline for the Tanzania agents task.

Fixes applied to the broken starter:

1. One embedding model for both sides. The starter encoded documents with
   all-MiniLM-L6-v2 and questions with e5-small-v2, so the two sets of vectors
   lived in unrelated spaces and their dot products were meaningless.
2. Correct e5 prefixes. e5 expects "query: " on questions and "passage: " on
   documents; the starter sent the query prefix to the wrong model and no
   passage prefix at all.
3. Whole documents instead of 180-character slices. Fixed-width slices cut
   sentences in half and separate the city and exhibit name (in the title)
   from the collection details, so a fragment of the wrong notice could win.
   Every document fits comfortably within the model's 512-token limit.
4. Only current access notices can be answers. Each exhibit and city has an
   ARCHIVED BACKGROUND notice that reads almost the same as the CURRENT
   ACCESS NOTICE but explicitly does not replace it. The questions ask for
   the current notice, so archived notices are ruled out after ranking.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Tazania.flag_utils import derive_flag, load_verification_codes

EMBEDDING_MODEL = "intfloat/e5-small-v2"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
CURRENT_NOTICE_PREFIX = "CURRENT ACCESS NOTICE"


@dataclass(frozen=True)
class Document:
    """One searchable source document."""

    document_id: str
    title: str
    text: str


def load_corpus(corpus_path: Path) -> list[Document]:
    """Read source documents from a JSON Lines corpus.

    :param corpus_path: Corpus file supplied with the task.
    :return: Searchable documents in file order.
    """
    documents: list[Document] = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line in corpus_file:
            row = json.loads(line)
            documents.append(Document(row["document_id"], row["title"], row["text"]))
    return documents


def load_questions(questions_path: Path) -> list[dict[str, str]]:
    """Read the ordered retrieval questions.

    :param questions_path: JSON question file supplied with the task.
    :return: Question objects in official order.
    """
    with questions_path.open(encoding="utf-8") as questions_file:
        questions = json.load(questions_file)
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a list.")
    return questions


def is_current_notice(document: Document) -> bool:
    """Tell whether a document is a current access notice.

    :param document: Source document.
    :return: True for current access notices, False for archived or general ones.
    """
    return document.title.upper().startswith(CURRENT_NOTICE_PREFIX)


def retrieve_documents(
    documents: list[Document],
    questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return one source document for every question.

    :param documents: Complete searchable source documents.
    :param questions: Ordered task questions.
    :return: Top source document ID for each question.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    document_vectors = encoder.encode(
        [f"{PASSAGE_PREFIX}{document.title}\n{document.text}" for document in documents],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_vectors = encoder.encode(
        [f"{QUERY_PREFIX}{question['question']}" for question in questions],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarities = np.asarray(query_vectors) @ np.asarray(document_vectors).T

    # Archived background notices and general bulletins never hold the
    # current collection details, so they cannot be the answer.
    current_mask = np.array([is_current_notice(document) for document in documents])
    if not current_mask.any():
        raise ValueError("The corpus contains no current access notices.")
    similarities[:, ~current_mask] = -np.inf

    results: list[dict[str, str]] = []
    for question, row in zip(questions, similarities):
        best_document_index = int(np.argmax(row))
        results.append(
            {
                "question_id": question["question_id"],
                "document_id": documents[best_document_index].document_id,
            }
        )
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line file locations.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("questions.json"))
    parser.add_argument("--output", type=Path, default=Path("retrieved_ids.json"))
    return parser.parse_args()


def main() -> None:
    """Run retrieval, save IDs, and print the derived flag."""
    arguments = parse_arguments()
    documents = load_corpus(arguments.corpus)
    questions = load_questions(arguments.questions)
    results = retrieve_documents(documents, questions)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    flag = derive_flag(results, questions, load_verification_codes(arguments.corpus))
    print(f"Saved {len(results)} results to {arguments.output}.")
    print(flag)


if __name__ == "__main__":
    main()
