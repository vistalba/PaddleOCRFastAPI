# -*- coding: utf-8 -*-
"""Shared PaddleOCR runtime configuration for API and process-pool workers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDX_CACHE_HOME = PROJECT_ROOT / ".paddlex"
DEFAULT_MODEL_TIER = "small"
SUPPORTED_MODEL_TIERS = {"tiny", "small", "medium"}
REQUIRED_MODEL_FILES = ("inference.json", "inference.pdiparams")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def configure_runtime() -> Path:
    """Set PaddleX flags before importing paddleocr/paddlex."""
    load_dotenv(PROJECT_ROOT / ".env")

    cache_home = _resolve_path(
        os.environ.get("PADDLE_PDX_CACHE_HOME", DEFAULT_PDX_CACHE_HOME)
    )
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_home)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
    return cache_home


@dataclass(frozen=True)
class OCRRuntimeConfig:
    model_tier: str
    device: str | None
    model_root: Path
    text_detection_model_name: str
    text_detection_model_dir: Path
    text_recognition_model_name: str
    text_recognition_model_dir: Path
    textline_orientation_model_name: str
    textline_orientation_model_dir: Path
    doc_orientation_model_name: str
    doc_orientation_model_dir: Path
    doc_unwarping_model_name: str
    doc_unwarping_model_dir: Path


def load_runtime_config() -> OCRRuntimeConfig:
    cache_home = configure_runtime()

    model_tier = os.environ.get("OCR_MODEL_TIER", DEFAULT_MODEL_TIER).strip().lower()
    if model_tier not in SUPPORTED_MODEL_TIERS:
        supported = ", ".join(sorted(SUPPORTED_MODEL_TIERS))
        raise ValueError(
            f"Unsupported OCR_MODEL_TIER={model_tier!r}，valid values are: {supported}"
        )

    raw_device = os.environ.get("OCR_DEVICE", "").strip()
    device = raw_device or None

    model_root = _resolve_path(
        os.environ.get("OCR_MODEL_DIR", cache_home / "official_models")
    )
    detection_name = f"PP-OCRv6_{model_tier}_det"
    recognition_name = f"PP-OCRv6_{model_tier}_rec"
    textline_name = "PP-LCNet_x1_0_textline_ori"
    doc_orientation_name = "PP-LCNet_x1_0_doc_ori"
    doc_unwarping_name = "UVDoc"

    def model_dir(env_name: str, model_name: str) -> Path:
        configured = os.environ.get(env_name)
        return _resolve_path(configured) if configured else model_root / model_name

    return OCRRuntimeConfig(
        model_tier=model_tier,
        device=device,
        model_root=model_root,
        text_detection_model_name=detection_name,
        text_detection_model_dir=model_dir(
            "OCR_TEXT_DETECTION_MODEL_DIR", detection_name
        ),
        text_recognition_model_name=recognition_name,
        text_recognition_model_dir=model_dir(
            "OCR_TEXT_RECOGNITION_MODEL_DIR", recognition_name
        ),
        textline_orientation_model_name=textline_name,
        textline_orientation_model_dir=model_dir(
            "OCR_TEXTLINE_ORIENTATION_MODEL_DIR", textline_name
        ),
        doc_orientation_model_name=doc_orientation_name,
        doc_orientation_model_dir=model_dir(
            "OCR_DOC_ORIENTATION_MODEL_DIR", doc_orientation_name
        ),
        doc_unwarping_model_name=doc_unwarping_name,
        doc_unwarping_model_dir=model_dir(
            "OCR_DOC_UNWARPING_MODEL_DIR", doc_unwarping_name
        ),
    )


def _validate_model_dir(model_name: str, model_dir: Path) -> None:
    missing = [
        filename
        for filename in REQUIRED_MODEL_FILES
        if not (model_dir / filename).is_file()
    ]
    if missing:
        missing_text = ", ".join(missing)
raise FileNotFoundError(
            f"Offline model {model_name} is incomplete: {model_dir} missing {missing_text}. "
            "Please run `uv run python -m scripts.prepare_models` in a connected environment, "
            "then copy the project directory to the target machine."
        )


def validate_ocr_models(config: OCRRuntimeConfig) -> None:
    _validate_model_dir(
        config.text_detection_model_name, config.text_detection_model_dir
    )
    _validate_model_dir(
        config.text_recognition_model_name, config.text_recognition_model_dir
    )
    _validate_model_dir(
        config.textline_orientation_model_name,
        config.textline_orientation_model_dir,
    )


def validate_doc_preprocessor_models(config: OCRRuntimeConfig) -> None:
    _validate_model_dir(
        config.doc_orientation_model_name, config.doc_orientation_model_dir
    )
    _validate_model_dir(
        config.doc_unwarping_model_name, config.doc_unwarping_model_dir
    )


def _device_kwargs(config: OCRRuntimeConfig) -> dict[str, Any]:
    if config.device is None:
        return {}

    if config.device.lower().startswith("gpu"):
        import paddle

if (
            not paddle.is_compiled_with_cuda()
            or paddle.device.cuda.device_count() == 0
        ):
            raise RuntimeError(
                f"OCR_DEVICE={device} but no Paddle CUDA GPU available in current environment. "
                "Windows target machine must have paddlepaddle-gpu installed"
            )

    return {"device": config.device}


def build_ocr():
    config = load_runtime_config()
    validate_ocr_models(config)

    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_detection_model_name=config.text_detection_model_name,
        text_detection_model_dir=str(config.text_detection_model_dir),
        text_recognition_model_name=config.text_recognition_model_name,
        text_recognition_model_dir=str(config.text_recognition_model_dir),
        textline_orientation_model_name=config.textline_orientation_model_name,
        textline_orientation_model_dir=str(config.textline_orientation_model_dir),
        **_device_kwargs(config),
    )


def build_doc_preprocessor():
    config = load_runtime_config()
    validate_doc_preprocessor_models(config)

    from paddleocr import DocPreprocessor

    return DocPreprocessor(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        doc_orientation_classify_model_name=config.doc_orientation_model_name,
        doc_orientation_classify_model_dir=str(config.doc_orientation_model_dir),
        doc_unwarping_model_name=config.doc_unwarping_model_name,
        doc_unwarping_model_dir=str(config.doc_unwarping_model_dir),
        **_device_kwargs(config),
    )
