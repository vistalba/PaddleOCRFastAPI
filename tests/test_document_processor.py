import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from utils.document_processor import (
    DocumentPreparationError,
    meaningful_text_char_count,
    normalize_pdf_text,
    prepare_document_file,
    prepare_document_files,
    should_use_native_pdf_text,
)


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def create_text_pdf(path: Path, page_texts: list[str]) -> None:
    page_count = len(page_texts)
    font_object_number = 3 + page_count * 2
    kids = " ".join(f"{3 + index * 2} 0 R" for index in range(page_count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"
        ).encode("ascii"),
    ]

    for index, text in enumerate(page_texts):
        content_number = 4 + index * 2
        content = (
            f"BT /F1 18 Tf 72 720 Td ({_pdf_escape(text)}) Tj ET"
        ).encode("latin-1")
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R "
                "/MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
                f"/Contents {content_number} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
            + content
            + b"\nendstream"
        )

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(obj)
        document.extend(b"\nendobj\n")

    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(document)


def create_scan_pdf(path: Path, page_count: int = 2) -> None:
    images = []
    for page_number in range(page_count):
        image = Image.new("RGB", (612, 792), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((60, 100, 550, 680), outline="black", width=4)
        draw.text((90, 160), f"Scanned page {page_number + 1}", fill="black")
        images.append(image)
    images[0].save(
        path,
        format="PDF",
        save_all=True,
        append_images=images[1:],
        resolution=96,
    )


def create_mixed_pdf(path: Path, text_pdf: Path, scan_pdf: Path) -> None:
    output = pdfium.PdfDocument.new()
    text_doc = pdfium.PdfDocument(str(text_pdf))
    scan_doc = pdfium.PdfDocument(str(scan_pdf))
    try:
        output.import_pages(text_doc, pages=[0])
        output.import_pages(scan_doc, pages=[0])
        output.save(str(path))
    finally:
        scan_doc.close()
        text_doc.close()
        output.close()


class DocumentProcessorTests(unittest.TestCase):
    def test_normalize_pdf_text(self):
        value = " First  line \r\n\r\n Second\tline\x00 \r Third "
        self.assertEqual(
            normalize_pdf_text(value),
            "First line\n\nSecond line\nThird",
        )

    def test_native_text_threshold(self):
        self.assertEqual(meaningful_text_char_count("页 1 / A"), 3)
        self.assertFalse(should_use_native_pdf_text("Page 1", minimum_chars=20))
        self.assertTrue(
            should_use_native_pdf_text(
                "This page contains enough native PDF text 12345",
                minimum_chars=20,
            )
        )

    def test_prepare_text_scan_and_mixed_pdfs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            text_task = root / "text_task"
            text_task.mkdir()
            text_pdf = text_task / "original.pdf"
            create_text_pdf(
                text_pdf,
                [
                    "Native text page one contains enough readable characters.",
                    "Native text page two also contains enough readable characters.",
                ],
            )
            text_result = prepare_document_file(str(text_pdf))
            self.assertEqual(text_result["file_type"], "pdf")
            self.assertEqual(
                [page["processing_method"] for page in text_result["pages"]],
                ["native_text", "native_text"],
            )
            self.assertTrue(
                all(Path(page["original_image_path"]).is_file() for page in text_result["pages"])
            )

            scan_task = root / "scan_task"
            scan_task.mkdir()
            scan_pdf = scan_task / "original.pdf"
            create_scan_pdf(scan_pdf)
            scan_result = prepare_document_file(str(scan_pdf))
            self.assertEqual(
                [page["processing_method"] for page in scan_result["pages"]],
                ["ocr", "ocr"],
            )

            mixed_task = root / "mixed_task"
            mixed_task.mkdir()
            mixed_pdf = mixed_task / "original.pdf"
            create_mixed_pdf(mixed_pdf, text_pdf, scan_pdf)
            mixed_result = prepare_document_file(str(mixed_pdf))
            self.assertEqual(
                [page["processing_method"] for page in mixed_result["pages"]],
                ["native_text", "ocr"],
            )
            self.assertEqual(
                [page["page_index"] for page in mixed_result["pages"]],
                [0, 1],
            )
            for page in mixed_result["pages"]:
                Path(page["original_image_path"]).resolve().relative_to(
                    mixed_task.resolve()
                )

    def test_prepare_image_as_single_page(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "original.png"
            Image.new("RGB", (120, 80), "white").save(image_path)

            result = prepare_document_file(str(image_path))

            self.assertEqual(result["file_type"], "image")
            self.assertEqual(len(result["pages"]), 1)
            self.assertEqual(result["pages"][0]["page_index"], 0)
            self.assertEqual(result["pages"][0]["processing_method"], "ocr")
            self.assertEqual(result["pages"][0]["width"], 120)
            self.assertEqual(result["pages"][0]["height"], 80)

    def test_prepare_multiple_images_keeps_upload_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.png"
            second = root / "second.png"
            Image.new("RGB", (120, 80), "red").save(first)
            Image.new("RGB", (90, 140), "blue").save(second)

            result = prepare_document_files(
                [str(second), str(first)]
            )

            self.assertEqual(result["file_type"], "image")
            self.assertEqual(
                [page["page_index"] for page in result["pages"]],
                [0, 1],
            )
            self.assertEqual(
                [Path(page["original_image_path"]).name for page in result["pages"]],
                ["second.png", "first.png"],
            )
            self.assertEqual(
                [(page["width"], page["height"]) for page in result["pages"]],
                [(90, 140), (120, 80)],
            )

            with patch(
                "utils.document_processor.MAX_MULTI_IMAGE_PAGES",
                1,
            ):
                with self.assertRaisesRegex(
                    DocumentPreparationError,
                    "exceeding maximum limit of 1",
                ):
                    prepare_document_files([str(first), str(second)])

    def test_pdf_page_limit_and_malformed_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_pdf = root / "two-pages.pdf"
            create_text_pdf(text_pdf, ["First page text.", "Second page text."])

            with patch("utils.document_processor.MAX_PDF_PAGES", 1):
                with self.assertRaisesRegex(
                    DocumentPreparationError,
                    "exceeding maximum limit of 1 pages",
                ):
                    prepare_document_file(str(text_pdf))

            malformed_pdf = root / "malformed.pdf"
            malformed_pdf.write_bytes(b"%PDF-1.4\nnot a complete PDF")
            with self.assertRaisesRegex(
                DocumentPreparationError,
                "Unable to open PDF",
            ):
                prepare_document_file(str(malformed_pdf))


if __name__ == "__main__":
    unittest.main()
