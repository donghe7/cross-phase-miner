"""Copy existing SQLite state into an empty PostgreSQL database.

Source is opened read-only, including its WAL. The target must contain no
CrossPhase data. DATABASE_URL is used so credentials need not appear in argv.
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path

from server.store import StateStore, is_postgres, learner_from_dict, model_from_dict


def read_snapshot(source: Path) -> dict:
    connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not {"intersections", "models", "learners"} <= tables:
            raise ValueError("Source is not a CrossPhase SQLite database")
        snapshot = {
            "intersections": connection.execute(
                "SELECT id, name, district, grid_x, grid_y, in_zone FROM intersections"
            ).fetchall()
        }
        for table in ("models", "learners", "red_measurements", "model_history"):
            snapshot[table] = (
                [
                    (iid, period, json.loads(payload))
                    for iid, period, payload in connection.execute(
                        f"SELECT id, period, payload FROM {table}"
                    )
                ]
                if table in tables
                else []
            )
        snapshot["visits"] = (
            [json.loads(row[0]) for row in connection.execute("SELECT payload FROM visits")]
            if "visits" in tables
            else []
        )
        snapshot["learning_jobs"] = (
            [
                (row[0], row[1], json.loads(row[2]), *row[3:])
                for row in connection.execute(
                    "SELECT record_id, id, payload, status, attempts, "
                    "next_attempt, created_at, last_error FROM learning_jobs"
                )
            ]
            if "learning_jobs" in tables
            else []
        )
        for _, _, payload in snapshot["models"]:
            model_from_dict(payload)
        for _, _, payload in snapshot["learners"]:
            learner_from_dict(payload)
        return snapshot
    finally:
        connection.close()


def migrate(source: Path, target: str) -> dict:
    if not is_postgres(target):
        raise ValueError("Set DATABASE_URL to a postgresql:// connection URL")
    snapshot = read_snapshot(source)
    store = StateStore(target)
    try:
        store.import_snapshot(snapshot)
    finally:
        store.close()
    return {table: len(rows) for table, rows in snapshot.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    counts = migrate(args.source, os.environ.get("DATABASE_URL", ""))
    print("Migration committed; source unchanged:", json.dumps(counts))


if __name__ == "__main__":
    main()
