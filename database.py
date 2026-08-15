# -*- coding: utf-8 -*-

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

# Run database migration before creating engine
if DATABASE_URL.startswith("sqlite"):
    from pathlib import Path
    db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
    if db_path.exists() or db_path.parent.exists():
        try:
            from database_migration import migrate_database
            migrate_database(str(db_path))
        except Exception as e:
            print(f"Warning: Database migration failed: {e}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
