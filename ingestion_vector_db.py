import os
import csv
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv("creds.env")

CSV_FILE        = "data/papers_20250101_20250107.csv"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "allenai-specter")
CHROMA_DIR      = "./chroma_db"
BATCH_SIZE      = 500


def get_arxiv_store(embeddings=None):
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name="arxiv_papers",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def get_user_store(embeddings=None):
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name="user_papers",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def load_from_csv(store):
    path = Path(CSV_FILE)
    if not path.exists():
        raise FileNotFoundError(f"{CSV_FILE} not found")

    existing = set(store.get()["ids"])

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    texts, metas, ids = [], [], []
    skipped = 0

    for row in rows:
        arxiv_id = row["arxiv_id"].strip()
        if arxiv_id in existing:
            skipped += 1
            continue

        text = f"{row['title'].strip()}. {row['abstract'].strip()}"
        metadata = {
            "arxiv_id":      arxiv_id,
            "title":         row["title"].strip(),
            "authors":       row["authors"].strip(),
            "categories":    row["categories"].strip(),
            "primary_cat":   row["primary_cat"].strip(),
            "published_at":  row["published_at"].strip(),
            "updated_at":    row["updated_at"].strip(),
            "pdf_url":       row["pdf_url"].strip(),
            "doi":           row["doi"].strip(),
            "source":        "arxiv",
            "has_full_text": "false",
        }

        texts.append(text)
        metas.append(metadata)
        ids.append(arxiv_id)

        if len(texts) == BATCH_SIZE:
            store.add_texts(texts=texts, metadatas=metas, ids=ids)
            existing.update(ids)
            print(f"inserted {len(ids)} | skipped {skipped}")
            texts, metas, ids = [], [], []

    if texts:
        store.add_texts(texts=texts, metadatas=metas, ids=ids)
        print(f"inserted {len(ids)} | skipped {skipped}")

    print(f"done → {CSV_FILE}")


def load_from_user(store, full_text, metadata):
    arxiv_id = metadata.get("arxiv_id", "")
    filename = metadata.get("filename", "")
    doc_id   = arxiv_id or filename or full_text[:60].replace(" ", "_")

    existing = set(store.get()["ids"])
    if doc_id in existing:
        print(f"already exists: {doc_id}")
        return

    store.add_texts(
        texts=[full_text],
        metadatas=[metadata],
        ids=[doc_id],
    )
    print(f"stored: {doc_id}")


if __name__ == "__main__":
    store = get_arxiv_store()
    load_from_csv(store)