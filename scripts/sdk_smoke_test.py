#!/usr/bin/env python3
"""
End-to-end smoke test using the real azure-ai-documentintelligence SDK.

This script replays exactly what Paperless-ngx's azureai remote OCR provider
does (see paperless-ngx ``src/paperless/parsers/remote.py``,
``RemoteDocumentParser._azure_ai_vision_parse``):

1. ``begin_analyze_document(model_id="prebuilt-read",
   body=AnalyzeDocumentRequest(bytes_source=...),
   output_content_format=TEXT, output=[PDF])``
   -> POST /documentintelligence/documentModels/prebuilt-read:analyze
      with a JSON ``{"base64Source": ...}`` body
2. ``poller.wait()``
   -> GET the Operation-Location until the status is terminal
3. ``result_id = poller.details["operation_id"]``
   -> the SDK parses the last path segment of the Operation-Location URL
4. ``result = poller.result()``
   -> the extracted text (``result.content``)
5. ``client.get_analyze_result_pdf(model_id="prebuilt-read",
   result_id=result_id)``
   -> GET /documentintelligence/documentModels/prebuilt-read/
      analyzeResults/{result_id}/pdf

If this script completes, the API is wire-compatible with the SDK that
Paperless-ngx uses.

Usage:
    python scripts/sdk_smoke_test.py <document.pdf|document.png> [endpoint]

The endpoint defaults to http://localhost:8000. The SDK requires a
credential; the API ignores it, so a dummy key is used.
"""

import sys
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.ai.documentintelligence.models import AnalyzeOutputOption
from azure.ai.documentintelligence.models import DocumentContentFormat
from azure.core.credentials import AzureKeyCredential


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <document.pdf|document.png> [endpoint]")
        sys.exit(1)

    document_path = Path(sys.argv[1])
    endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"

    if not document_path.is_file():
        print(f"Document not found: {document_path}")
        sys.exit(1)

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential("unused"),
    )

    try:
        print(f"Submitting {document_path} to {endpoint} ...")
        with document_path.open("rb") as f:
            analyze_request = AnalyzeDocumentRequest(bytes_source=f.read())
            poller = client.begin_analyze_document(
                model_id="prebuilt-read",
                body=analyze_request,
                output_content_format=DocumentContentFormat.TEXT,
                output=[AnalyzeOutputOption.PDF],
                content_type="application/json",
            )

        poller.wait()
        result_id = poller.details["operation_id"]
        result = poller.result()

        print(f"operation_id: {result_id}")
        text = result.content or ""
        print(f"Extracted {len(text)} characters of text")
        preview = text[:200].replace("\n", " | ")
        print(f"Preview: {preview!r}")

        archive_path = document_path.with_name(f"{document_path.stem}_sdk_archive.pdf")
        with archive_path.open("wb") as f:
            for chunk in client.get_analyze_result_pdf(
                model_id="prebuilt-read",
                result_id=result_id,
            ):
                f.write(chunk)
        print(f"Archive PDF saved to: {archive_path}")
        print("OK: full SDK flow (submit -> poll -> text -> archive PDF) succeeded")
    finally:
        client.close()


if __name__ == "__main__":
    main()
