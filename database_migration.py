# -*- coding: utf-8 -*-
"""Database migration script to add new columns to task_pages table."""

import sqlite3
from pathlib import Path


def migrate_database(db_path: str) -> None:
    """Add missing columns to task_pages table if they don't exist."""
    path = Path(db_path)
    if not path.exists():
        # Database doesn't exist yet, will be created fresh
        return

    try:
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        # Check if columns exist
        cursor.execute("PRAGMA table_info(task_pages)")
        columns = {row[1] for row in cursor.fetchall()}

        # Columns to add
        columns_to_add = [
            ("pdf_width_pts", "FLOAT"),
            ("pdf_height_pts", "FLOAT"),
            ("render_scale", "FLOAT"),
        ]

        # Add missing columns
        for col_name, col_type in columns_to_add:
            if col_name not in columns:
                cursor.execute(
                    f"ALTER TABLE task_pages ADD COLUMN {col_name} {col_type}"
                )
                print(f"Added column: {col_name}")

        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Database migration error: {e}")
        raise