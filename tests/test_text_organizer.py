import json
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from schema_migrations import ensure_task_page_organization_columns
from utils.text_organizer import (
    TextLine,
    TextOrganizerError,
    _organize_chunk_with_ai,
    _parse_ai_groups,
    build_rule_result,
    join_text_lines,
)


def _ocr_result(texts, boxes):
    return [
        {
            "rec_texts": texts,
            "rec_scores": [0.98] * len(texts),
            "rec_boxes": boxes,
            "rec_polys": [],
        }
    ]


class CoordinateRuleTests(unittest.TestCase):
    def test_heading_close_lines_and_large_gap_form_paragraphs(self):
        result = build_rule_result(
            processing_method="ocr",
            native_text=None,
            ocr_result=_ocr_result(
                ["文档标题", "这是第一行", "这是第二行。", "新段落"],
                [
                    [250, 10, 550, 40],
                    [70, 60, 730, 80],
                    [70, 84, 730, 104],
                    [70, 145, 730, 165],
                ],
            ),
            page_width=800,
        )

        self.assertEqual(
            [segment["text"] for segment in result["segments"]],
            ["文档标题", "这是第一行这是第二行。", "新段落"],
        )
        self.assertEqual(
            result["segments"][1]["source_line_indices"],
            [1, 2],
        )

    def test_two_columns_are_read_column_by_column(self):
        result = build_rule_result(
            processing_method="ocr",
            native_text=None,
            ocr_result=_ocr_result(
                ["左一", "右一", "左二", "右二", "左三", "右三"],
                [
                    [50, 20, 330, 40],
                    [470, 20, 750, 40],
                    [50, 45, 330, 65],
                    [470, 45, 750, 65],
                    [50, 70, 330, 90],
                    [470, 70, 750, 90],
                ],
            ),
            page_width=800,
        )

        self.assertEqual(
            [
                segment["source_line_indices"]
                for segment in result["segments"]
            ],
            [[0, 2, 4], [1, 3, 5]],
        )
        self.assertEqual(
            [segment["text"] for segment in result["segments"]],
            ["左一左二左三", "右一右二右三"],
        )

    def test_native_blank_lines_are_preserved_as_paragraphs(self):
        result = build_rule_result(
            processing_method="native_text",
            native_text="first line\nsecond line\n\nnext paragraph",
            ocr_result=None,
        )
        self.assertEqual(
            [segment["text"] for segment in result["segments"]],
            ["first line second line", "next paragraph"],
        )

    def test_list_items_keep_a_line_break_when_grouped(self):
        self.assertEqual(
            join_text_lines(
                [
                    TextLine(0, "说明："),
                    TextLine(1, "1. 第一项"),
                    TextLine(2, "2. 第二项"),
                ]
            ),
            "说明：\n1. 第一项\n2. 第二项",
        )


class AIResultValidationTests(unittest.TestCase):
    def test_ai_must_preserve_every_line_id_in_order(self):
        self.assertEqual(
            _parse_ai_groups(
                '{"paragraphs":[[4,7],[9]]}',
                [4, 7, 9],
            ),
            [[4, 7], [9]],
        )
        with self.assertRaises(TextOrganizerError):
            _parse_ai_groups(
                '{"paragraphs":[[4,9],[7]]}',
                [4, 7, 9],
            )

    def test_ai_only_controls_grouping_not_text(self):
        class FakeModel:
            def create_chat_completion(self, **_kwargs):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"paragraphs": [[0, 1], [2]]}
                                )
                            }
                        }
                    ]
                }

        lines = [
            TextLine(0, "原文A"),
            TextLine(1, "原文B"),
            TextLine(2, "原文C"),
        ]
        groups = _organize_chunk_with_ai(FakeModel(), lines)
        self.assertEqual(
            [[line.text for line in group] for group in groups],
            [["原文A", "原文B"], ["原文C"]],
        )


class SchemaMigrationTests(unittest.TestCase):
    def test_existing_task_pages_table_gets_organization_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = create_engine(
                f"sqlite:///{Path(directory) / 'old.db'}"
            )
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "CREATE TABLE task_pages ("
                        "id INTEGER PRIMARY KEY, "
                        "task_id VARCHAR(36) NOT NULL, "
                        "page_index INTEGER NOT NULL)"
                    )
                )

            ensure_task_page_organization_columns(engine)
            columns = {
                column["name"]
                for column in inspect(engine).get_columns("task_pages")
            }
            self.assertTrue(
                {
                    "rule_result",
                    "ai_result",
                    "ai_status",
                    "ai_model_name",
                    "ai_processed_at",
                    "ai_error",
                }.issubset(columns)
            )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
