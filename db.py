"""
SQLAlchemy logging DB phase final piece.

Several fields are nullable because RAG doesn't exist yet (retrieved_doc_id, retrieval_score, grounding_passed) — the
schema is built wide now so no migration is needed later.
"""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, String, Float, Boolean, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

DATABASE_URL = "sqlite:///logs.db"
engine = create_engine(DATABASE_URL, echo=False)


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    latency_ms: Mapped[float] = mapped_column(Float)
    user_query: Mapped[str] = mapped_column(String)
    category: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    intent: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    branch: Mapped[str] = mapped_column(String)  # "direct" | "rag" — always "direct" until "rag" is implemented
    retrieved_doc_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)   # RAG only
    retrieval_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # RAG only
    valid_json: Mapped[bool] = mapped_column(Boolean)
    in_taxonomy: Mapped[bool] = mapped_column(Boolean)
    grounding_passed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # RAG only
    guardrail_outcome: Mapped[str] = mapped_column(String)
    fallback_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    raw_output: Mapped[str] = mapped_column(String)


def init_db() -> None:
    """Idempotent — safe to call on every app startup."""
    Base.metadata.create_all(engine)


def log_request(**kwargs) -> None:
    with Session(engine) as session:
        entry = RequestLog(**kwargs)
        session.add(entry)
        session.commit()