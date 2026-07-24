# -*- coding: utf-8 -*-
"""Small in-place schema migrations for installations without Alembic."""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def ensure_task_page_organization_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    if "task_pages" not in inspector.get_table_names():
        return

    existing = {
        column["name"]
        for column in inspector.get_columns("task_pages")
    }
    long_text = "LONGTEXT" if engine.dialect.name == "mysql" else "TEXT"
    definitions = {
        "rule_result": long_text,
        "ai_result": long_text,
        "ai_status": (
            "VARCHAR(20) NOT NULL DEFAULT 'not_started'"
        ),
        "ai_model_name": "VARCHAR(255)",
        "ai_processed_at": "DATETIME",
        "ai_error": "TEXT",
    }
    with engine.begin() as connection:
        for column_name, ddl in definitions.items():
            if column_name in existing:
                continue
            connection.execute(
                text(
                    f"ALTER TABLE task_pages "
                    f"ADD COLUMN {column_name} {ddl}"
                )
            )
