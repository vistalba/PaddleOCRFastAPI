# -*- coding: utf-8 -*-
"""Download the selected PP-OCRv6 models into the project-local PaddleX cache."""

from __future__ import annotations

import argparse

from utils.ocr_runtime import (
    load_runtime_config,
    validate_doc_preprocessor_models,
    validate_ocr_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载当前 OCR_MODEL_TIER 对应的离线模型"
    )
    parser.add_argument(
        "--with-doc-preprocessor",
        action="store_true",
        help="同时准备文档方向分类与图像矫正模型",
    )
    args = parser.parse_args()

    config = load_runtime_config()

    from paddleocr import DocPreprocessor, PaddleOCR

    print(
        f"准备 PP-OCRv6_{config.model_tier} 模型，"
        f"缓存目录: {config.model_root}"
    )
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_detection_model_name=config.text_detection_model_name,
        text_recognition_model_name=config.text_recognition_model_name,
        textline_orientation_model_name=config.textline_orientation_model_name,
        device=config.device,
    )
    ocr.close()
    validate_ocr_models(config)

    if args.with_doc_preprocessor:
        preprocessor = DocPreprocessor(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            doc_orientation_classify_model_name=config.doc_orientation_model_name,
            doc_unwarping_model_name=config.doc_unwarping_model_name,
            device=config.device,
        )
        preprocessor.close()
        validate_doc_preprocessor_models(config)

    print("模型准备完成，可将项目目录复制到离线目标机。")


if __name__ == "__main__":
    main()
