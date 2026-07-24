# -*- coding: utf-8 -*-

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ocr_tasks.db")
UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./uploads"))
RATE_LIMIT: str = os.getenv("RATE_LIMIT", "10/minute")
# 同时处理的 OCR 任务上限；其余任务排入 queued 等待队列
MAX_CONCURRENT_OCR: int = int(os.getenv("MAX_CONCURRENT_OCR", "1"))
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "50"))
PDF_RENDER_SCALE: float = float(os.getenv("PDF_RENDER_SCALE", "2.0"))
PDF_MAX_RENDER_PIXELS: int = int(
    os.getenv("PDF_MAX_RENDER_PIXELS", "40000000")
)
PDF_NATIVE_TEXT_MIN_CHARS: int = int(
    os.getenv("PDF_NATIVE_TEXT_MIN_CHARS", "20")
)
