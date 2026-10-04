"""Rewrites the artifact locations stored in an MLflow SQLite database so they
point at a different mlruns/ root. Used when building the Docker image:
mlflow.db records absolute paths from the machine it was trained on (e.g.
file:///C:/Users/.../mlruns), which don't exist inside the Linux container.

Usage: python scripts/rewrite_artifact_paths.py <db_path> <new_mlruns_uri>
"""

import re
import sqlite3
import sys

db_path, new_root = sys.argv[1], sys.argv[2].rstrip("/")
pattern = re.compile(r"^file:///.*?/mlruns")

conn = sqlite3.connect(db_path)
TARGETS = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
]
for table, column in TARGETS:
    rows = conn.execute(f"SELECT rowid, {column} FROM {table}").fetchall()
    updated = 0
    for rowid, value in rows:
        if value and pattern.match(value):
            conn.execute(f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
                         (pattern.sub(new_root, value), rowid))
            updated += 1
    print(f"{table}.{column}: rewrote {updated} of {len(rows)} rows")
conn.commit()
conn.close()
