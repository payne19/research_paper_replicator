import os
import re
import csv
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv("creds.env")

CSV_FILE        = "data/papers_20250101_20250107.csv"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "allenai-specter")
CHROMA_DIR      = "./chroma_db"
BATCH_SIZE      = 500

CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"(https?://\S+)", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()


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

        text = clean_text(f"{row['title'].strip()}. {row['abstract'].strip()}")
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
    chunk_ids = [i for i in existing if i.startswith(doc_id)]
    if chunk_ids:
        print(f"already exists: {doc_id}")
        return

    full_text = clean_text(full_text)
    chunks    = splitter.split_text(full_text)

    texts, metas, ids = [], [], []
    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metas.append({**metadata, "chunk_index": str(i), "total_chunks": str(len(chunks))})
        ids.append(f"{doc_id}_chunk_{i}")

    for i in range(0, len(texts), BATCH_SIZE):
        store.add_texts(
            texts=texts[i : i + BATCH_SIZE],
            metadatas=metas[i : i + BATCH_SIZE],
            ids=ids[i : i + BATCH_SIZE],
        )
        print(f"stored chunks {i} to {min(i + BATCH_SIZE, len(texts))} of {len(texts)}")

    print(f"done: {len(chunks)} chunks stored for {doc_id}")


if __name__ == "__main__":
    store = get_arxiv_store()
    load_from_csv(store)