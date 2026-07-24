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
MAX_MULTI_IMAGE_PAGES: int = int(os.getenv("MAX_MULTI_IMAGE_PAGES", "50"))
RETRY_COOLDOWN_SECONDS: int = int(
    os.getenv("RETRY_COOLDOWN_SECONDS", "60")
)
PDF_RENDER_SCALE: float = float(os.getenv("PDF_RENDER_SCALE", "2.0"))
PDF_MAX_RENDER_PIXELS: int = int(
    os.getenv("PDF_MAX_RENDER_PIXELS", "40000000")
)
PDF_NATIVE_TEXT_MIN_CHARS: int = int(
    os.getenv("PDF_NATIVE_TEXT_MIN_CHARS", "20")
)

# 可选的本地 GGUF 文本整理模型。未配置路径时仅启用坐标规则整理。
AI_TEXT_MODEL_PATH: str = os.getenv("AI_TEXT_MODEL_PATH", "").strip()
AI_TEXT_MODEL_NAME: str = os.getenv("AI_TEXT_MODEL_NAME", "").strip()
AI_TEXT_MODEL_CHAT_FORMAT: str = os.getenv(
    "AI_TEXT_MODEL_CHAT_FORMAT", ""
).strip()
AI_TEXT_MODEL_CONTEXT_SIZE: int = max(
    512,
    int(os.getenv("AI_TEXT_MODEL_CONTEXT_SIZE", "4096")),
)
AI_TEXT_MODEL_THREADS: int = max(
    1,
    int(
        os.getenv(
            "AI_TEXT_MODEL_THREADS",
            str(max(1, min(8, (os.cpu_count() or 2) // 2))),
        )
    ),
)
AI_TEXT_MODEL_GPU_LAYERS: int = int(
    os.getenv("AI_TEXT_MODEL_GPU_LAYERS", "0")
)
AI_TEXT_MODEL_MAX_TOKENS: int = max(
    128,
    int(os.getenv("AI_TEXT_MODEL_MAX_TOKENS", "768")),
)
AI_TEXT_MODEL_SEED: int = int(os.getenv("AI_TEXT_MODEL_SEED", "2026"))
AI_TEXT_MAX_LINES_PER_CHUNK: int = max(
    10,
    int(os.getenv("AI_TEXT_MAX_LINES_PER_CHUNK", "80")),
)
AI_TEXT_MAX_INPUT_CHARS: int = max(
    1000,
    int(os.getenv("AI_TEXT_MAX_INPUT_CHARS", "7000")),
)
