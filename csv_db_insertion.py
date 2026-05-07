import csv
import psycopg2
from pathlib import Path
from setup_db import get_conn

CSV_FILE   = "data/papers_20250101_20250107.csv"
BATCH_SIZE = 500


def parse(row):
    return {
        "arxiv_id":    row["arxiv_id"].strip(),
        "title":       row["title"].strip(),
        "abstract":    row["abstract"].strip(),
        "authors":     [a.strip() for a in row["authors"].split("|") if a.strip()],
        "categories":  [c.strip() for c in row["categories"].split("|") if c.strip()],
        "primary_cat": row["primary_cat"].strip(),
        "published_at": row["published_at"].strip() or None,
        "updated_at":  row["updated_at"].strip() or None,
        "pdf_url":     row["pdf_url"].strip() or None,
        "doi":         row["doi"].strip() or None,
    }


def load_batch(cur, batch):
    for paper in batch:
        cur.execute("""
            INSERT INTO papers
                (arxiv_id, title, abstract, categories, primary_cat,
                 published_at, updated_at, pdf_url, doi)
            VALUES
                (%(arxiv_id)s, %(title)s, %(abstract)s, %(categories)s, %(primary_cat)s,
                 %(published_at)s, %(updated_at)s, %(pdf_url)s, %(doi)s)
            ON CONFLICT (arxiv_id) DO NOTHING
            RETURNING id;
        """, paper)

        row = cur.fetchone()
        if not row:
            continue
        paper_id = row[0]

        for pos, name in enumerate(paper["authors"]):
            cur.execute(
                "INSERT INTO authors (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id;",
                (name,)
            )
            result = cur.fetchone()
            if result:
                author_id = result[0]
            else:
                cur.execute("SELECT id FROM authors WHERE name = %s;", (name,))
                author_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO paper_authors (paper_id, author_id, position) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                (paper_id, author_id, pos)
            )


def main():
    path = Path(CSV_FILE)
    if not path.exists():
        raise FileNotFoundError(f"{CSV_FILE} not found")

    with open(path, "r", encoding="utf-8") as f:
        papers = [parse(row) for row in csv.DictReader(f)]

    conn = get_conn()
    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i : i + BATCH_SIZE]
        with conn:
            with conn.cursor() as cur:
                load_batch(cur, batch)

    conn.close()
    print(f"loaded {len(papers)} papers from {CSV_FILE}")


if __name__ == "__main__":
    main()