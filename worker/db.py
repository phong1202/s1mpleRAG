"""SYNC engine, deliberately different from the API's async one.

Two engines, one set of models. A Celery task is a plain `def` -- no
asyncio.run() anywhere in worker/.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

engine = create_engine(get_settings().worker_database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, class_=Session, expire_on_commit=False)


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
