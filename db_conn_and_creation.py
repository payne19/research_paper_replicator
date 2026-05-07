import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

HOST     = "localhost"
PORT     = 5432
DB_NAME  = "arxiv_intel"
USER     = "postgres"
PASSWORD = "postgres"


def get_conn(dbname=DB_NAME):
    return psycopg2.connect(host=HOST, port=PORT, dbname=dbname, user=USER, password=PASSWORD)


def create_db():
    conn = psycopg2.connect(host=HOST, port=PORT, dbname="postgres", user=USER, password=PASSWORD)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {DB_NAME}")
    conn.close()


def create_tables():
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id           SERIAL PRIMARY KEY,
            arxiv_id     TEXT UNIQUE NOT NULL,
            title        TEXT NOT NULL,
            abstract     TEXT NOT NULL,
            categories   TEXT[],
            primary_cat  TEXT,
            published_at TIMESTAMPTZ,
            updated_at   TIMESTAMPTZ,
            pdf_url      TEXT,
            doi          TEXT,
            ingested_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS authors (
            id   SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_authors (
            paper_id  INT REFERENCES papers(id)  ON DELETE CASCADE,
            author_id INT REFERENCES authors(id) ON DELETE CASCADE,
            position  SMALLINT,
            PRIMARY KEY (paper_id, author_id)
        );

        CREATE INDEX IF NOT EXISTS idx_papers_published   ON papers (published_at DESC);
        CREATE INDEX IF NOT EXISTS idx_papers_primary_cat ON papers (primary_cat);
        CREATE INDEX IF NOT EXISTS idx_pa_author          ON paper_authors (author_id);
    """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_db()
    create_tables()