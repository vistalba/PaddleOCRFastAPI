# -*- coding: utf-8 -*-

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ocr_tasks.db")
UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./uploads"))
RATE_LIMIT: str = os.getenv("RATE_LIMIT", "10/minute")
# Maximum concurrent OCR tasks; remaining tasks are queued
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

# OCR rendering DPI (dots per inch) for image-based OCR
# Higher DPI = better accuracy but slower processing
# Range: 72-600, Default: 300 (recommended by PaddleOCR)
OCR_DPI: int = max(72, min(600, int(os.getenv("OCR_DPI", "300"))))

# Optional local GGUF text organizer model. When not configured, only coordinate-based rule organization is enabled.
AI_TEXT_MODEL_PATH: str = os.getenv("AI_TEXT_MODEL_PATH", "").strip()
AI_TEXT_MODEL_NAME: str = os.getenv("AI_TEXT_MODEL_NAME", "").strip()
AI_TEXT_MODEL_CHAT_FORMAT: str = os.getenv(
    "AI_TEXT_MODEL_CHAT_FORMAT", ""
).strip()
AI_TEXT_ORGANIZER_MODE: str = os.getenv(
    "AI_TEXT_ORGANIZER_MODE", "compare"
).strip().lower()
if AI_TEXT_ORGANIZER_MODE not in {"page", "boundary", "compare"}:
    AI_TEXT_ORGANIZER_MODE = "compare"
AI_TEXT_MODEL_CONTEXT_SIZE: int = max(
    512,
    int(os.getenv("AI_TEXT_MODEL_CONTEXT_SIZE", "4096")),
)
AI_TEXT_MODEL_THREADS: int = max(
    1,
    int(
        int(os.getenv("AI_TEXT_MODEL_THREADS", "0")) or max(1, min(8, (os.cpu_count() or 2) // 2))
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
AI_TEXT_BOUNDARY_MERGE_THRESHOLD: float = min(
    1.0,
    max(
        0.5,
        float(os.getenv("AI_TEXT_BOUNDARY_MERGE_THRESHOLD", "0.72")),
    ),
)
AI_TEXT_BOUNDARY_SPLIT_THRESHOLD: float = min(
    0.5,
    max(
        0.0,
        float(os.getenv("AI_TEXT_BOUNDARY_SPLIT_THRESHOLD", "0.20")),
    ),
)
AI_TEXT_BOUNDARY_CONTEXT_CHARS: int = max(
    100,
    int(os.getenv("AI_TEXT_BOUNDARY_CONTEXT_CHARS", "600")),
)
AI_TEXT_MAX_LINES_PER_CHUNK: int = max(
    10,
    int(os.getenv("AI_TEXT_MAX_LINES_PER_CHUNK", "80")),
)
AI_TEXT_MAX_INPUT_CHARS: int = max(
    1000,
    int(os.getenv("AI_TEXT_MAX_INPUT_CHARS", "7000")),
)
TEXT_RULE_FONT_HEIGHT_RATIO: float = max(
    1.05,
    float(os.getenv("TEXT_RULE_FONT_HEIGHT_RATIO", "1.55")),
)
TEXT_RULE_TITLE_BODY_HEIGHT_RATIO: float = max(
    1.05,
    float(os.getenv("TEXT_RULE_TITLE_BODY_HEIGHT_RATIO", "1.18")),
)
TEXT_RULE_LINE_STEP_RATIO: float = max(
    1.05,
    float(os.getenv("TEXT_RULE_LINE_STEP_RATIO", "2.0")),
)
TEXT_RULE_HORIZONTAL_GAP_RATIO: float = max(
    1.0,
    float(os.getenv("TEXT_RULE_HORIZONTAL_GAP_RATIO", "3.0")),
)

# Force OCR for all PDF processing (even if PDF has native text)
FORCE_OCR_FOR_AZURE: bool = os.getenv("FORCE_OCR_FOR_AZURE", "false").lower() == "true"