# -*- coding: utf-8 -*-
"""Turn OCR lines into stable paragraphs, optionally assisted by a local GGUF LLM."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import (
    AI_TEXT_MAX_INPUT_CHARS,
    AI_TEXT_MAX_LINES_PER_CHUNK,
    AI_TEXT_MODEL_CHAT_FORMAT,
    AI_TEXT_MODEL_CONTEXT_SIZE,
    AI_TEXT_MODEL_GPU_LAYERS,
    AI_TEXT_MODEL_MAX_TOKENS,
    AI_TEXT_MODEL_NAME,
    AI_TEXT_MODEL_PATH,
    AI_TEXT_MODEL_SEED,
    AI_TEXT_MODEL_THREADS,
)

ORGANIZATION_RESULT_VERSION = 1
_TERMINAL_PUNCTUATION = frozenset("。！？!?；;…」』”’）)]】")
_NO_SPACE_BEFORE = frozenset(
    "，。！？；：、,.!?;:%％)]}）】》〉」』”’"
)
_NO_SPACE_AFTER = frozenset("([{（【《〈「『“‘")
_LIST_PREFIX = re.compile(
    r"^\s*(?:"
    r"[-•·▪◦]\s*|"
    r"\d{1,3}[.、．)]\s*|"
    r"[A-Za-z][.)]\s*|"
    r"[（(]?[一二三四五六七八九十百]+[、.)）]\s*"
    r")"
)
_HEADING_PREFIX = re.compile(
    r"^\s*(?:"
    r"第[一二三四五六七八九十百千万\d]+[章节篇部分]|"
    r"[一二三四五六七八九十]+、|"
    r"\d{1,3}(?:\.\d{1,3}){0,3}\s+"
    r")"
)
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


class TextOrganizerError(RuntimeError):
    """Raised when the optional AI organizer cannot produce a safe result."""


@dataclass(frozen=True)
class TextLine:
    source_index: int
    text: str
    box: tuple[float, float, float, float] | None = None
    score: float | None = None
    paragraph_before: bool = False
    column: int = 0

    @property
    def left(self) -> float:
        return self.box[0] if self.box else 0.0

    @property
    def top(self) -> float:
        return self.box[1] if self.box else float(self.source_index)

    @property
    def right(self) -> float:
        return self.box[2] if self.box else self.left

    @property
    def bottom(self) -> float:
        return self.box[3] if self.box else self.top

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _box_from_values(
    box_value: Any,
    poly_value: Any,
) -> tuple[float, float, float, float] | None:
    if isinstance(box_value, (list, tuple)) and len(box_value) >= 4:
        values = [_finite_number(value) for value in box_value[:4]]
        if all(value is not None for value in values):
            left, top, right, bottom = values
            if right >= left and bottom >= top:
                return left, top, right, bottom

    if not isinstance(poly_value, (list, tuple)):
        return None
    points = []
    for point in poly_value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = _finite_number(point[0])
        y = _finite_number(point[1])
        if x is not None and y is not None:
            points.append((x, y))
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def extract_ocr_lines(ocr_result: Any) -> list[TextLine]:
    """Flatten PaddleOCR results while retaining indices used by the frontend."""
    lines: list[TextLine] = []
    source_index = 0
    if not isinstance(ocr_result, list):
        return lines

    for item in ocr_result:
        if not isinstance(item, dict):
            continue
        texts = item.get("rec_texts")
        if not isinstance(texts, list):
            continue
        scores = item.get("rec_scores")
        boxes = item.get("rec_boxes")
        polys = item.get("rec_polys")
        scores = scores if isinstance(scores, list) else []
        boxes = boxes if isinstance(boxes, list) else []
        polys = polys if isinstance(polys, list) else []

        for item_index, value in enumerate(texts):
            text = str(value).strip() if value is not None else ""
            score = (
                _finite_number(scores[item_index])
                if item_index < len(scores)
                else None
            )
            box = _box_from_values(
                boxes[item_index] if item_index < len(boxes) else None,
                polys[item_index] if item_index < len(polys) else None,
            )
            if text:
                lines.append(
                    TextLine(
                        source_index=source_index,
                        text=text,
                        box=box,
                        score=score,
                    )
                )
            source_index += 1
    return lines


def extract_native_text_lines(value: str | None) -> list[TextLine]:
    lines: list[TextLine] = []
    paragraph_pending = False
    source_index = 0
    for raw_line in (value or "").replace("\r\n", "\n").replace("\r", "\n").split(
        "\n"
    ):
        text = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not text:
            if lines:
                paragraph_pending = True
            continue
        lines.append(
            TextLine(
                source_index=source_index,
                text=text,
                paragraph_before=paragraph_pending,
            )
        )
        source_index += 1
        paragraph_pending = False
    return lines


def extract_page_lines(
    processing_method: str,
    native_text: str | None,
    ocr_result: Any,
) -> list[TextLine]:
    if processing_method == "native_text":
        return extract_native_text_lines(native_text)
    return extract_ocr_lines(ocr_result)


def _sort_band(
    lines: Sequence[TextLine],
    midpoint: float,
    two_columns: bool,
) -> list[TextLine]:
    if not two_columns:
        return [
            replace(line, column=0)
            for line in sorted(lines, key=lambda item: (item.top, item.left))
        ]
    left = sorted(
        (line for line in lines if line.center_x < midpoint),
        key=lambda item: (item.top, item.left),
    )
    right = sorted(
        (line for line in lines if line.center_x >= midpoint),
        key=lambda item: (item.top, item.left),
    )
    return [
        *(replace(line, column=0) for line in left),
        *(replace(line, column=1) for line in right),
    ]


def order_text_lines(
    lines: Sequence[TextLine],
    page_width: int | float | None = None,
) -> list[TextLine]:
    """Recover a conservative single/two-column reading order."""
    if not lines:
        return []
    coordinate_lines = [line for line in lines if line.box is not None]
    if len(coordinate_lines) != len(lines):
        return list(lines)

    content_left = min(line.left for line in coordinate_lines)
    content_right = max(line.right for line in coordinate_lines)
    effective_width = max(
        1.0,
        float(page_width or 0),
        content_right,
    )
    content_width = max(1.0, content_right - content_left)
    midpoint = content_left + content_width / 2

    narrow = [
        line
        for line in coordinate_lines
        if line.width <= content_width * 0.62
    ]
    left_count = sum(line.center_x < midpoint for line in narrow)
    right_count = sum(line.center_x >= midpoint for line in narrow)
    two_columns = (
        len(coordinate_lines) >= 6
        and left_count >= 2
        and right_count >= 2
        and content_width >= effective_width * 0.55
    )
    if not two_columns:
        return _sort_band(coordinate_lines, midpoint, False)

    spanning = sorted(
        (
            line
            for line in coordinate_lines
            if line.width >= content_width * 0.68
            or (line.left < midpoint < line.right)
        ),
        key=lambda item: (item.top, item.left),
    )
    non_spanning = [line for line in coordinate_lines if line not in spanning]

    ordered: list[TextLine] = []
    band_top = float("-inf")
    for span in spanning:
        band = [
            line
            for line in non_spanning
            if band_top <= line.top < span.top
        ]
        ordered.extend(_sort_band(band, midpoint, True))
        ordered.append(replace(span, column=-1))
        non_spanning = [line for line in non_spanning if line not in band]
        band_top = span.bottom
    ordered.extend(_sort_band(non_spanning, midpoint, True))
    return ordered


def _is_cjk_character(value: str) -> bool:
    return bool(value and _CJK.fullmatch(value))


def join_text_lines(lines: Sequence[TextLine]) -> str:
    """Join line text without allowing a model to rewrite OCR characters."""
    result = ""
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        if not result:
            result = text
            continue
        previous = result[-1]
        first = text[0]
        if _LIST_PREFIX.match(text):
            result += "\n" + text
        elif previous == "-" and previous.isascii() and first.isascii():
            result = result[:-1] + text
        elif (
            previous in _NO_SPACE_AFTER
            or first in _NO_SPACE_BEFORE
            or _is_cjk_character(previous)
            or _is_cjk_character(first)
        ):
            result += text
        else:
            result += " " + text
    return result


def _median(values: Iterable[float], default: float) -> float:
    filtered = [value for value in values if math.isfinite(value)]
    return statistics.median(filtered) if filtered else default


def _looks_like_list_item(text: str) -> bool:
    return bool(_LIST_PREFIX.match(text))


def _looks_like_heading(
    line: TextLine,
    median_height: float,
    content_left: float,
    content_right: float,
) -> bool:
    compact_text = re.sub(r"\s+", "", line.text)
    if not compact_text or len(compact_text) > 32:
        return False
    if _HEADING_PREFIX.match(line.text):
        return True
    if compact_text[-1] in _TERMINAL_PUNCTUATION or compact_text[-1] in "，,：:":
        return False
    if line.height > median_height * 1.28:
        return True
    content_width = max(1.0, content_right - content_left)
    centered = abs(line.center_x - (content_left + content_right) / 2)
    return (
        line.box is not None
        and line.width < content_width * 0.66
        and centered < content_width * 0.12
        and len(compact_text) <= 22
    )


def _should_start_paragraph(
    previous: TextLine,
    current: TextLine,
    median_height: float,
    gap_threshold: float,
    common_left: dict[int, float],
    previous_is_heading: bool,
    current_is_heading: bool,
) -> bool:
    if current.paragraph_before:
        return True
    if previous.column != current.column:
        return True
    if previous_is_heading or current_is_heading:
        return True

    current_is_list = _looks_like_list_item(current.text)
    previous_is_list = _looks_like_list_item(previous.text)
    if current_is_list:
        return True
    if previous_is_list and not current_is_list:
        return True

    if previous.box is None or current.box is None:
        return False
    vertical_gap = current.top - previous.bottom
    if vertical_gap > gap_threshold:
        return True

    baseline_left = common_left.get(current.column, current.left)
    indent = current.left - baseline_left
    previous_ended = previous.text.rstrip()[-1:] in _TERMINAL_PUNCTUATION
    if indent > max(12.0, median_height * 0.9) and previous_ended:
        return True
    if previous_ended and vertical_gap > median_height * 0.36:
        return True
    return False


def _result_from_groups(
    groups: Sequence[Sequence[TextLine]],
    method: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    segments = []
    for group in groups:
        text = join_text_lines(group)
        if not text:
            continue
        segments.append(
            {
                "text": text,
                "source_line_indices": [
                    line.source_index for line in group
                ],
            }
        )
    result: dict[str, Any] = {
        "version": ORGANIZATION_RESULT_VERSION,
        "method": method,
        "segments": segments,
        "text": "\n\n".join(segment["text"] for segment in segments),
    }
    if model_name:
        result["model_name"] = model_name
    return result


def build_rule_result(
    processing_method: str,
    native_text: str | None,
    ocr_result: Any,
    page_width: int | float | None = None,
) -> dict[str, Any]:
    lines = order_text_lines(
        extract_page_lines(processing_method, native_text, ocr_result),
        page_width=page_width,
    )
    if not lines:
        return _result_from_groups([], "coordinate_rules")
    if all(line.box is None for line in lines):
        if processing_method == "native_text":
            groups: list[list[TextLine]] = []
            for line in lines:
                if not groups or line.paragraph_before:
                    groups.append([line])
                else:
                    groups[-1].append(line)
            return _result_from_groups(groups, "native_paragraphs")
        return _result_from_groups(
            [[line] for line in lines],
            "coordinate_rules_fallback",
        )
    if any(line.box is None for line in lines):
        return _result_from_groups(
            [[line] for line in lines],
            "coordinate_rules_fallback",
        )

    heights = [line.height for line in lines if line.height > 0]
    median_height = _median(heights, 16.0)
    same_column_gaps = [
        current.top - previous.bottom
        for previous, current in zip(lines, lines[1:])
        if previous.column == current.column
        and current.top >= previous.top
        and current.top - previous.bottom >= 0
    ]
    median_gap = _median(same_column_gaps, median_height * 0.2)
    gap_threshold = max(
        median_height * 0.72,
        median_gap * 2.2 + 1.0,
    )
    content_left = min(line.left for line in lines)
    content_right = max(line.right for line in lines)
    common_left = {
        column: _median(
            (line.left for line in lines if line.column == column),
            content_left,
        )
        for column in {line.column for line in lines}
    }

    groups = [[lines[0]]]
    previous_heading = _looks_like_heading(
        lines[0], median_height, content_left, content_right
    )
    for current in lines[1:]:
        previous = groups[-1][-1]
        current_heading = _looks_like_heading(
            current, median_height, content_left, content_right
        )
        if _should_start_paragraph(
            previous,
            current,
            median_height,
            gap_threshold,
            common_left,
            previous_heading,
            current_heading,
        ):
            groups.append([current])
        else:
            groups[-1].append(current)
        previous_heading = current_heading
    return _result_from_groups(groups, "coordinate_rules")


def build_raw_result(
    processing_method: str,
    native_text: str | None,
    ocr_result: Any,
) -> dict[str, Any]:
    lines = extract_page_lines(processing_method, native_text, ocr_result)
    return _result_from_groups([[line] for line in lines], "raw_lines")


def ai_organizer_status() -> dict[str, Any]:
    model_path = Path(AI_TEXT_MODEL_PATH).expanduser() if AI_TEXT_MODEL_PATH else None
    model_name = AI_TEXT_MODEL_NAME or (model_path.stem if model_path else None)
    if model_path is None:
        return {
            "available": False,
            "model_name": model_name,
            "reason": "未配置 AI_TEXT_MODEL_PATH",
        }
    if model_path.suffix.lower() != ".gguf":
        return {
            "available": False,
            "model_name": model_name,
            "reason": "AI 模型必须是 GGUF 文件",
        }
    if not model_path.is_file():
        return {
            "available": False,
            "model_name": model_name,
            "reason": "配置的 AI 模型文件不存在",
        }
    if importlib.util.find_spec("llama_cpp") is None:
        return {
            "available": False,
            "model_name": model_name,
            "reason": "未安装可选依赖 llama-cpp-python",
        }
    return {
        "available": True,
        "model_name": model_name,
        "reason": None,
    }


_llm_instance: Any = None
_llm_signature: tuple[Any, ...] | None = None
_llm_lock = threading.Lock()


def _get_llm() -> Any:
    global _llm_instance, _llm_signature
    status = ai_organizer_status()
    if not status["available"]:
        raise TextOrganizerError(status["reason"])

    model_path = str(Path(AI_TEXT_MODEL_PATH).expanduser().resolve())
    signature = (
        model_path,
        AI_TEXT_MODEL_CONTEXT_SIZE,
        AI_TEXT_MODEL_THREADS,
        AI_TEXT_MODEL_GPU_LAYERS,
        AI_TEXT_MODEL_CHAT_FORMAT,
    )
    with _llm_lock:
        if _llm_instance is not None and _llm_signature == signature:
            return _llm_instance
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise TextOrganizerError(
                "未安装可选依赖 llama-cpp-python"
            ) from exc

        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": AI_TEXT_MODEL_CONTEXT_SIZE,
            "n_threads": AI_TEXT_MODEL_THREADS,
            "n_threads_batch": AI_TEXT_MODEL_THREADS,
            "n_batch": min(512, AI_TEXT_MODEL_CONTEXT_SIZE),
            "n_gpu_layers": AI_TEXT_MODEL_GPU_LAYERS,
            "seed": AI_TEXT_MODEL_SEED,
            "verbose": False,
        }
        if AI_TEXT_MODEL_CHAT_FORMAT:
            kwargs["chat_format"] = AI_TEXT_MODEL_CHAT_FORMAT
        try:
            _llm_instance = Llama(**kwargs)
        except Exception as exc:
            raise TextOrganizerError(f"AI 模型加载失败：{exc}") from exc
        _llm_signature = signature
        return _llm_instance


def _chunk_lines(lines: Sequence[TextLine]) -> list[list[TextLine]]:
    chunks: list[list[TextLine]] = []
    current: list[TextLine] = []
    current_chars = 0
    for line in lines:
        line_chars = len(line.text) + 48
        if current and (
            len(current) >= AI_TEXT_MAX_LINES_PER_CHUNK
            or current_chars + line_chars > AI_TEXT_MAX_INPUT_CHARS
        ):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(line)
        current_chars += line_chars
    if current:
        chunks.append(current)
    return chunks


def _prompt_payload(lines: Sequence[TextLine]) -> str:
    payload = []
    for prompt_index, line in enumerate(lines):
        item: dict[str, Any] = {
            "id": prompt_index,
            "text": line.text,
        }
        if line.box is not None:
            item["bbox"] = [round(value, 1) for value in line.box]
            item["column"] = line.column
        payload.append(item)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _parse_ai_groups(
    content: str,
    expected_ids: Sequence[int],
) -> list[list[int]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TextOrganizerError("AI 未返回有效 JSON") from exc
    paragraphs = payload.get("paragraphs") if isinstance(payload, dict) else None
    if not isinstance(paragraphs, list) or not paragraphs:
        raise TextOrganizerError("AI 结果缺少 paragraphs")

    groups: list[list[int]] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, list) or not paragraph:
            raise TextOrganizerError("AI 返回了空段落")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in paragraph):
            raise TextOrganizerError("AI 段落编号格式无效")
        groups.append(paragraph)
    flattened = [value for group in groups for value in group]
    if flattened != list(expected_ids):
        raise TextOrganizerError("AI 改变、遗漏或重复了原始行编号")
    return groups


def _organize_chunk_with_ai(
    llm: Any,
    lines: Sequence[TextLine],
) -> list[list[TextLine]]:
    expected_ids = list(range(len(lines)))
    system_prompt = (
        "你是 OCR 文档分段器。你只能根据文字、坐标和阅读顺序决定段落边界，"
        "不得修改、补充、删除或重排行。输出严格 JSON："
        '{"paragraphs":[[行号,行号],[行号]]}。'
        "每个输入行号必须恰好出现一次，且顺序必须与输入完全一致。"
        "标题、列表项通常独立成段；正文中属于同一语义段落的连续行应合并。"
    )
    user_prompt = "请整理以下 OCR 行：\n" + _prompt_payload(lines)
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            top_p=0.8,
            max_tokens=AI_TEXT_MODEL_MAX_TOKENS,
            seed=AI_TEXT_MODEL_SEED,
        )
        choice = response["choices"][0]
        message = choice.get("message") if isinstance(choice, dict) else None
        content = (
            message.get("content")
            if isinstance(message, dict)
            else choice.get("text")
        )
    except Exception as exc:
        raise TextOrganizerError(f"AI 推理失败：{exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise TextOrganizerError("AI 未返回整理结果")

    groups = _parse_ai_groups(content.strip(), expected_ids)
    return [[lines[prompt_index] for prompt_index in group] for group in groups]


def build_ai_result(
    processing_method: str,
    native_text: str | None,
    ocr_result: Any,
    page_width: int | float | None = None,
) -> dict[str, Any]:
    """Run the configured local model and return validated paragraph groups."""
    lines = order_text_lines(
        extract_page_lines(processing_method, native_text, ocr_result),
        page_width=page_width,
    )
    if not lines:
        raise TextOrganizerError("当前页没有可整理的文字")
    llm = _get_llm()
    groups: list[list[TextLine]] = []
    for chunk in _chunk_lines(lines):
        groups.extend(_organize_chunk_with_ai(llm, chunk))
    status = ai_organizer_status()
    return _result_from_groups(
        groups,
        "local_ai",
        model_name=status["model_name"],
    )
