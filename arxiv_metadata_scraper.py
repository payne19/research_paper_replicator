import arxiv
import csv
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("creds.env")

NEW_DATA_ONLY = os.getenv("NEW_DATA", "TRUE").upper() == "TRUE"

if NEW_DATA_ONLY:
    START_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    END_DATE   = datetime.now().strftime("%Y-%m-%d")
else:
    START_DATE = os.getenv("START_DATE")
    END_DATE   = os.getenv("END_DATE", datetime.now().strftime("%Y-%m-%d"))

CATEGORIES = [
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "cs.IR",
    "cs.NE",
    "cs.MA",
    "cs.DB",
    "cs.ET",
    "stat.ML",
]

OUTPUT_DIR       = Path("data")
CHUNK_DAYS       = int(os.getenv("CHUNK_DAYS", 7))
PAGE_SIZE        = int(os.getenv("PAGE_SIZE", 100))
API_DELAY        = int(os.getenv("API_DELAY", 3))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", 1000))

FIELDS = [
    "arxiv_id", "title", "abstract", "authors",
    "categories", "primary_cat", "published_at",
    "updated_at", "pdf_url", "doi",
]


def to_dt(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def end_of_day(d):
    return d.replace(hour=23, minute=59, second=59)


def chunks(start, end):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        yield cur, end_of_day(chunk_end)
        cur = chunk_end + timedelta(days=1)


def arxiv_range(start, end):
    fmt = "%Y%m%d%H%M%S"
    return f"[{start.strftime(fmt)} TO {end.strftime(fmt)}]"


def fetch(start, end):
    cats   = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    query  = f"({cats}) AND submittedDate:{arxiv_range(start, end)}"
    client = arxiv.Client(page_size=PAGE_SIZE, delay_seconds=API_DELAY, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    rows = []
    for r in client.results(search):
        arxiv_id = r.entry_id.split("/abs/")[-1].split("v")[0]
        rows.append({
            "arxiv_id":    arxiv_id,
            "title":       r.title.strip().replace("\n", " "),
            "abstract":    r.summary.strip().replace("\n", " "),
            "authors":     "|".join(a.name for a in r.authors),
            "categories":  "|".join(r.categories),
            "primary_cat": r.primary_category,
            "published_at": r.published.isoformat(),
            "updated_at":  r.updated.isoformat() if r.updated else "",
            "pdf_url":     r.pdf_url or "",
            "doi":         r.doi or "",
        })
    return rows


def append_to_csv(rows, path):
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)


def main():
    start = to_dt(START_DATE)
    end   = to_dt(END_DATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"papers_{START_DATE.replace('-','')}_{END_DATE.replace('-','')}.csv"

    if out_path.exists():
        out_path.unlink()

    buffer      = []
    total_saved = 0
    seen_ids    = set()

    for chunk_start, chunk_end in chunks(start, end):
        rows = fetch(chunk_start, chunk_end)

        for row in rows:
            if row["arxiv_id"] not in seen_ids:
                seen_ids.add(row["arxiv_id"])
                buffer.append(row)

        if len(buffer) >= CHECKPOINT_EVERY:
            append_to_csv(buffer, out_path)
            total_saved += len(buffer)
            print(f"checkpoint: {total_saved} rows saved")
            buffer = []

    if buffer:
        append_to_csv(buffer, out_path)
        total_saved += len(buffer)

    print(f"done: {total_saved} papers → {out_path}")


if __name__ == "__main__":
    main()