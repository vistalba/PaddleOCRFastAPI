# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class OCRModel(BaseModel):
    input_path: Optional[str] = None
    rec_texts: List[str] = []
    rec_scores: List[float] = []
    rec_polys: List[Any] = []
    rec_boxes: List[Any] = []


class Base64PostModel(BaseModel):
    base64_str: str  # Base64 string
