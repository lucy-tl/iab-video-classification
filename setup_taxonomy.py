"""
One-time setup: load IAB Content Taxonomy 3.1 TSV into SQLite.

Run this before using any pipeline. The DB is read into memory at pipeline
startup and used to resolve parent chains (T1–T4) from matched node IDs.

Usage:
    python3 setup_taxonomy.py --load-tsv Content_Taxonomy_3.1.tsv
    python3 setup_taxonomy.py --inspect   # verify DB stats after loading

The TSV is available at:
    https://github.com/InteractiveAdvertisingBureau/Taxonomies
"""

import sqlite3
import csv
import argparse

DB_PATH = "iab_taxonomy_3.1.db"


def load_tsv_to_db(tsv_path: str, db_path: str = DB_PATH):
    """Parse IAB 3.1 TSV and load into SQLite."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS iab_taxonomy")
    cur.execute("""
        CREATE TABLE iab_taxonomy (
            unique_id       TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            parent_id       TEXT,
            tier            INTEGER,
            full_path       TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON iab_taxonomy(name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_parent ON iab_taxonomy(parent_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tier ON iab_taxonomy(tier)")

    rows_loaded = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        next(f)  # skip title row
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            unique_id = row.get("Unique ID", "").strip()
            name      = row.get("Name", "").strip()
            parent_id = row.get("Parent", "").strip() or None
            if not unique_id or not name:
                continue
            cur.execute(
                "INSERT OR REPLACE INTO iab_taxonomy (unique_id, name, parent_id, tier) VALUES (?, ?, ?, ?)",
                (unique_id, name, parent_id, 0)
            )
            rows_loaded += 1

    # Assign tiers by parent relationship
    cur.execute("UPDATE iab_taxonomy SET tier = 1 WHERE parent_id IS NULL OR parent_id = ''")
    cur.execute("""
        UPDATE iab_taxonomy SET tier = 2
        WHERE parent_id IN (SELECT unique_id FROM iab_taxonomy WHERE tier = 1)
    """)
    cur.execute("""
        UPDATE iab_taxonomy SET tier = 3
        WHERE parent_id IN (SELECT unique_id FROM iab_taxonomy WHERE tier = 2)
    """)
    cur.execute("""
        UPDATE iab_taxonomy SET tier = 4
        WHERE parent_id IN (SELECT unique_id FROM iab_taxonomy WHERE tier = 3)
    """)

    # Build full_path for each tier
    cur.execute("UPDATE iab_taxonomy SET full_path = name WHERE tier = 1")
    cur.execute("""
        UPDATE iab_taxonomy SET full_path = (
            SELECT t1.name || ' > ' || t2.name
            FROM iab_taxonomy t1
            JOIN iab_taxonomy t2 ON t2.unique_id = iab_taxonomy.unique_id
            WHERE t1.unique_id = iab_taxonomy.parent_id
        )
        WHERE tier = 2
    """)
    cur.execute("""
        UPDATE iab_taxonomy SET full_path = (
            SELECT t1.name || ' > ' || t2.name || ' > ' || t3.name
            FROM iab_taxonomy t1
            JOIN iab_taxonomy t2 ON t2.parent_id = t1.unique_id
            JOIN iab_taxonomy t3 ON t3.unique_id = iab_taxonomy.unique_id
            WHERE t2.unique_id = iab_taxonomy.parent_id
              AND t1.unique_id = t2.parent_id
        )
        WHERE tier = 3
    """)
    cur.execute("""
        UPDATE iab_taxonomy SET full_path = (
            SELECT t1.name || ' > ' || t2.name || ' > ' || t3.name || ' > ' || t4.name
            FROM iab_taxonomy t1
            JOIN iab_taxonomy t2 ON t2.parent_id = t1.unique_id
            JOIN iab_taxonomy t3 ON t3.parent_id = t2.unique_id
            JOIN iab_taxonomy t4 ON t4.unique_id = iab_taxonomy.unique_id
            WHERE t3.unique_id = iab_taxonomy.parent_id
              AND t2.unique_id = t3.parent_id
              AND t1.unique_id = t2.parent_id
        )
        WHERE tier = 4
    """)

    conn.commit()
    conn.close()
    print(f"✅ Loaded {rows_loaded} rows into {db_path}")
    return rows_loaded


def inspect_db(db_path: str = DB_PATH):
    """Print DB stats and sample rows."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for tier in [1, 2, 3, 4]:
        cur.execute("SELECT COUNT(*) FROM iab_taxonomy WHERE tier = ?", (tier,))
        print(f"Tier {tier}: {cur.fetchone()[0]} rows")

    print("\nSample Tier 1 rows:")
    cur.execute("SELECT unique_id, name FROM iab_taxonomy WHERE tier = 1 LIMIT 5")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    print("\nSample Sports T2/T3/T4:")
    cur.execute("""
        SELECT t.unique_id, t.name, t.tier, t.full_path
        FROM iab_taxonomy t
        JOIN iab_taxonomy t1 ON (
            t.parent_id = t1.unique_id
            OR t.parent_id IN (SELECT unique_id FROM iab_taxonomy WHERE parent_id = t1.unique_id)
            OR t.parent_id IN (SELECT unique_id FROM iab_taxonomy WHERE parent_id IN
                (SELECT unique_id FROM iab_taxonomy WHERE parent_id = t1.unique_id))
        )
        WHERE t1.name = 'Sports' AND t1.tier = 1
        LIMIT 10
    """)
    for row in cur.fetchall():
        print(f"  [T{row[2]}] {row[0]}: {row[3]}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-time setup: load IAB 3.1 taxonomy TSV into SQLite")
    parser.add_argument("--load-tsv", metavar="PATH", help="Path to Content_Taxonomy_3.1.tsv")
    parser.add_argument("--inspect",  action="store_true", help="Print DB stats")
    parser.add_argument("--db",       default=DB_PATH, help=f"DB path (default: {DB_PATH})")
    args = parser.parse_args()

    if args.load_tsv:
        load_tsv_to_db(args.load_tsv, args.db)
    if args.inspect:
        inspect_db(args.db)
    if not args.load_tsv and not args.inspect:
        parser.print_help()
