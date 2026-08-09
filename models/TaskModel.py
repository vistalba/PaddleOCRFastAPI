# -*- coding: utf-8 -*-

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship

from database import Base


class Task(Base):
    __tablename__ = "tasks"

    task_id = Column(String(36), primary_key=True, index=True)
    ip = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="queued")  # queued / processing / done / failed
    original_filename = Column(String(255), nullable=True)
    file_dir = Column(String(512), nullable=True)
    use_doc_preprocessor = Column(Boolean, nullable=False, default=False)
    force_ocr = Column(Boolean, nullable=False, default=False)
    ocr_result = Column(Text, nullable=True)   # JSON string
    error_msg = Column(Text, nullable=True)

    pages = relationship(
        "TaskPage",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="TaskPage.page_index",
    )


large_text_type = Text().with_variant(LONGTEXT(), "mysql")


class TaskPage(Base):
    __tablename__ = "task_pages"
    __table_args__ = (
        UniqueConstraint("task_id", "page_index", name="uq_task_pages_task_page"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(
        String(36),
        ForeignKey("tasks.task_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_index = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="queued")
    processing_method = Column(String(20), nullable=False, default="ocr")
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    original_image_path = Column(String(512), nullable=True)
    corrected_image_path = Column(String(512), nullable=True)
    native_text = Column(large_text_type, nullable=True)
    # PDF metadata for coordinate transformation
    pdf_width_pts = Column(Integer, nullable=True)  # PDF width in points (72 DPI)
    pdf_height_pts = Column(Integer, nullable=True)  # PDF height in points
    render_scale = Column(Float, nullable=True)  # Scale factor used for rendering
    ocr_result = Column(large_text_type, nullable=True)
    rule_result = Column(large_text_type, nullable=True)
    ai_result = Column(large_text_type, nullable=True)
    ai_status = Column(
        String(20),
        nullable=False,
        default="not_started",
    )
    ai_model_name = Column(String(255), nullable=True)
    ai_processed_at = Column(DateTime, nullable=True)
    ai_error = Column(Text, nullable=True)
    error_msg = Column(Text, nullable=True)

    task = relationship("Task", back_populates="pages")
