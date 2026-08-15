# Azure AI Protocol Parsing Framework - Implementation Plan

## Project Overview

This document outlines the implementation of an Azure Document Intelligence-compatible API layer for PaddleOCRFastAPI, enabling integration with Paperless-ngx and other systems that expect Azure's Document Intelligence protocol.

### Repository Configuration
- **Source Repository**: https://github.com/vistalba/PaddleOCRFastAPI
- **Source Branch**: `english-translation`
- **Target Feature Branch**: `feature/azureai-compatibility`
- **Docker Registry**: Docker Hub (`vistalba/paddleocrfastapi`)

---

## Implementation Phases

### Phase 1: Branch Isolation & CI/CD Automation ✓

**Status**: ✅ COMPLETED

**Deliverables**:
1. Created `.github/workflows/build-test-image.yml`
2. Configured multi-platform Docker builds (linux/amd64, linux/arm64)
3. Set up automatic triggering on `feature/azureai-compatibility` branch

**Workflow Configuration**:
- **Trigger**: Push to `feature/azureai-compatibility` + manual dispatch
- **Image Tags**:
  - `vistalba/paddleocrfastapi:test-azureai` (fixed tag)
  - `vistalba/paddleocrfastapi:feature-azureai-compatibility-{short-sha}` (unique per commit)
- **Build Strategy**: Docker Buildx with QEMU for multi-platform support
- **Cache**: GitHub Actions cache for faster builds

**Required Secrets**:
- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token

---

### Phase 2: Application Dependencies ✓

**Status**: ✅ COMPLETED

**File Modified**: `pyproject.toml`

**Added Dependencies**:
```toml
"reportlab>=4.1.0",    # PDF canvas layer creation
"pypdf>=4.3.0",        # PDF manipulation and merging
```

**Note**: `uv.lock` will be automatically regenerated during Docker build.

---

### Phase 3: Azure API Protocol Translator ✓

**Status**: ✅ COMPLETED

**File Created**: `azure_api.py`

**Module Components**:

#### 1. Helper Functions

**`create_searchable_pdf_layer()`**
- Translates PaddleOCR bounding boxes to PDF coordinate system
- Creates invisible text layer overlay using ReportLab
- Merges with original PDF page using pypdf
- Returns modified PDF bytes with searchable text layer

#### 2. Azure-Compatible Endpoints

**`POST /documentintelligence/documentModels/prebuilt-layout:analyze`**
- **Purpose**: Accept PDF and schedule OCR task
- **Input**: Multipart form with PDF file
- **Output**: `202 Accepted` with `Operation-Location` header
- **Behavior**: Forwards to internal PaddleOCR `/ocr/tasks` endpoint

**`GET /documentintelligence/operations/{task_id}`**
- **Purpose**: Poll task status and retrieve results
- **Input**: Task ID from Operation-Location
- **Output**: Azure-compatible JSON with status and results
- **Status Mapping**:
  - `queued/processing` → `{"status": "running"}`
  - `failed` → `{"status": "failed"}`
  - `done` → Full Azure JSON with `analyzeResult`

**`GET /documentintelligence/operations/{task_id}/pdf`**
- **Purpose**: Download searchable PDF with embedded text layer
- **Input**: Task ID
- **Output**: PDF binary with `Content-Disposition` header
- **Error**: `404` if PDF not ready

#### 3. Task Storage

- Uses shared in-memory storage (`TASK_STORAGE`)
- Maps task IDs to:
  - Original PDF bytes
  - Azure-compatible JSON results
  - Searchable PDF archive
  - Creation timestamp

---

### Phase 4: Router Integration ✓

**Status**: ✅ COMPLETED

**File Modified**: `main.py`

**Changes**:
1. Added import: `from azure_api import router as azure_compatibility_router`
2. Registered router: `app.include_router(azure_compatibility_router)`

**Result**: Azure endpoints now accessible at root level:
- `/documentintelligence/documentModels/prebuilt-layout:analyze`
- `/documentintelligence/operations/{task_id}`
- `/documentintelligence/operations/{task_id}/pdf`

---

### Phase 5: SDK Wire Compatibility (azure-ai-documentintelligence 1.0.2) ✓

**Status**: ✅ COMPLETED

**Background**: The original layer was verified against the exact request contract of
the `azure-ai-documentintelligence==1.0.2` Python SDK (used by Paperless-ngx's
`azureai` remote OCR provider; paperless-ngx's `uv.lock` pins it together with
`azure-core==1.38.0`). The SDK's call sequence is:

1. `POST {endpoint}/documentintelligence/documentModels/prebuilt-read:analyze?api-version=2024-11-30&outputContentFormat=text&output=pdf`
   with a **JSON** body `{"base64Source": "<base64>"}` (`Content-Type: application/json`)
2. `GET` the `Operation-Location` URL until `status` is `succeeded`/`failed`
3. `GET {endpoint}/documentintelligence/documentModels/prebuilt-read/analyzeResults/{result_id}/pdf`
   where `result_id` is the last path segment of the `Operation-Location` URL

**Changes** (all in `azure_api.py`):

1. **Generic model id**: `POST /documentintelligence/documentModels/{model_id}:analyze`
   accepts any model id (Paperless-ngx sends `prebuilt-read`; the previous
   hard-coded `prebuilt-layout` route returned 404 for it)
2. **JSON `base64Source` body** (SDK format) in addition to the legacy
   multipart `file` upload
3. **Magic-byte file type detection** (PDF, PNG, JPEG, TIFF, BMP, GIF, WebP) -
   the SDK sends no file name, so the type is derived from the content
4. **New archive route**: `GET /documentintelligence/documentModels/{model_id}/analyzeResults/{result_id}/pdf`
   (the path the SDK calls; `result_id` == task id). The legacy
   `GET /documentintelligence/operations/{task_id}/pdf` is kept
5. **Image input support**: for image tasks the searchable archive PDF is
   rebuilt from the OCR'd page images (1 pixel == 1 PDF point, so OCR boxes in
   pixels map 1:1) with the invisible text layer overlaid
6. **Result fidelity**: `analyzeResult` now includes `stringIndexType:
   "textElements"`, echoes the requested `modelId`, reports image page
   dimensions with `unit: "pixel"`, and failed operations return an Azure-style
   `error: {code, message}` object
7. **Azure-style error payloads** `{"error": {"code", "message"}}` for 400/404/
   502 instead of opaque 500s; upstream 4xx/429 from the internal task API are
   propagated with their status code
8. **pypdf future-proofing**: pages are attached to a `PdfWriter` before
   content-stream mutation (mutating reader pages is deprecated in pypdf 6.x
   and removed in 7.0)

**Known external blocker (Paperless-ngx side)**:

Paperless-ngx's `RemoteDocumentParser._azure_ai_vision_parse` reads
`poller.details["operation_id"]`. `LROPoller.details` does not exist in any
azure-core release (verified 1.20 - 1.38), so production raises
`AttributeError` before the PDF download. Their unit tests mock the attribute
(`mock_poller.details = {"operation_id": "fake-op-id"}`), so their CI does not
catch it. This project waits for the upstream fix; until then, end-to-end runs
through Paperless-ngx fail at that line regardless of this API's
compatibility. The API itself can be verified directly with the SDK client
(`DocumentIntelligenceClient.begin_analyze_document(model_id="prebuilt-read",
document=...)` + `get_analyze_result_pdf`).

**Unit tests**: `tests/test_azure_api.py` (17 tests; the internal PaddleOCR API
is stubbed in-process, so no OCR models or running server are required):

```bash
uv run pytest tests/test_azure_api.py
```

Covered: SDK-format submit (PDF + image), legacy multipart submit, poll
running/succeeded/failed, `analyzeResult` fidelity (modelId, stringIndexType,
units, lines), archive PDF via both routes (dimensions + extractable text
layer), FORCE_OCR text-layer replacement, corrected-variant image fetch, and
all 400/404 error paths.

---

## Testing & Validation

### Step 1: Deploy Feature Branch

```bash
# Checkout feature branch
git checkout -b feature/azureai-compatibility english-translation

# Verify all files are in place
git status

# Commit changes
git add .github/workflows/build-test-image.yml
git add azure_api.py
git add main.py
git add pyproject.toml

git commit -m "feat: implement Azure Document Intelligence compatibility layer"

# Push to remote
git push origin feature/azureai-compatibility
```

### Step 2: Monitor GitHub Actions

1. Navigate to repository Actions tab
2. Verify workflow triggers automatically
3. Monitor build progress for multi-platform image
4. Confirm successful push to Docker Hub with tags:
   - `vistalba/paddleocrfastapi:test-azureai`
   - `vistalba/paddleocrfastapi:feature-azureai-compatibility-{sha}`

### Step 3: Deploy Test Container

**Docker Compose Configuration**:
```yaml
services:
  PaddleOCR-Testing:
    image: vistalba/paddleocrfastapi:test-azureai
    container_name: paddle_ocr_azureai_test
    environment:
      - TZ=Europe/Zurich
      - OCR_MODEL_TIER=small
      - OCR_DEVICE=cpu
    ports:
      - "8000:8000"
    restart: unless-stopped
```

**Deploy**:
```bash
docker compose up -d
```

### Step 4: API Testing

#### Test 1: Submit PDF for Analysis (SDK format)

```bash
BASE64=$(base64 -w 0 test-document.pdf)
curl -X POST "http://localhost:8000/documentintelligence/documentModels/prebuilt-read:analyze?api-version=2024-11-30" \
  -H "Content-Type: application/json" \
  -d "{\"base64Source\": \"$BASE64\"}"
```

**Expected Response**: `202 Accepted`
```
Operation-Location: http://localhost:8000/documentintelligence/operations/{task_id}?api-version=2024-11-30
```

The legacy multipart format also works (any model id):

```bash
curl -X POST "http://localhost:8000/documentintelligence/documentModels/prebuilt-read:analyze?api-version=2024-11-30" \
  -F "file=@test-document.pdf"
```

#### Test 2: Poll Task Status

```bash
curl "http://localhost:8000/documentintelligence/operations/{task_id}?api-version=2024-11-30"
```

**Expected Response** (while processing):
```json
{"status": "running"}
```

**Expected Response** (when complete):
```json
{
  "status": "succeeded",
  "createdDateTime": "2026-08-08T16:00:00Z",
  "lastUpdatedDateTime": "2026-08-08T16:01:00Z",
  "analyzeResult": {
    "apiVersion": "2024-11-30",
    "modelId": "prebuilt-read",
    "stringIndexType": "textElements",
    "content": "Full extracted text...",
    "pages": [
      {
        "pageNumber": 1,
        "angle": 0,
        "width": 8.5,
        "height": 11,
        "unit": "inch",
        "lines": [
          {"content": "Line 1 text"},
          {"content": "Line 2 text"}
        ]
      }
    ]
  }
}
```

#### Test 3: Download Searchable PDF

```bash
# SDK path (result_id == task_id)
curl "http://localhost:8000/documentintelligence/documentModels/prebuilt-read/analyzeResults/{task_id}/pdf" \
  -o output-searchable.pdf

# Legacy path (still supported)
curl "http://localhost:8000/documentintelligence/operations/{task_id}/pdf" \
  -o output-searchable.pdf
```

**Expected**: PDF file with embedded searchable text layer

### Step 5: Paperless-ngx Integration

**Configuration**:
```env
PAPERLESS_OCR_LANGUAGE=eng
PAPERLESS_OCR_BACKEND=tesseract
# Point to Azure-compatible endpoint
PAPERLESS_REMOTE_OCR_ENDPOINT=https://your-domain.com
```

**Verification**:
1. Upload test PDF to Paperless-ngx
2. Verify text extraction completes
3. Open document and test text selection/highlighting
4. Search for words within the document

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    External Client (Paperless-ngx)              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Azure Document Intelligence API
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Azure Compatibility Layer (azure_api.py)           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ POST /documentintelligence/documentModels/...:analyze   │  │
│  │ GET  /documentintelligence/operations/{task_id}         │  │
│  │ GET  /documentintelligence/operations/{task_id}/pdf     │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Internal PaddleOCR API
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PaddleOCR Core (routers/tasks.py)            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ POST /ocr/tasks           - Schedule OCR task            │  │
│  │ GET  /ocr/tasks/{task_id} - Query task status/results   │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ OCR Processing
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PaddleOCR Engine                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ PP-OCRv6_small            - Detection & Recognition      │  │
│  │ Document Preprocessor     - Image correction             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
PaddleOCRFastAPI/
├── .github/
│   └── workflows/
│       └── build-test-image.yml          # NEW: CI/CD workflow
├── azure_api.py                          # NEW: Azure compatibility layer
├── main.py                               # MODIFIED: Added Azure router
├── pyproject.toml                        # MODIFIED: Added dependencies + pytest config
├── tests/
│   └── test_azure_api.py                 # NEW: Unit tests (stubbed PaddleOCR backend)
├── scripts/
│   └── analyze_boxes.py                  # MODIFIED: SDK-format submit + image support
├── Dockerfile                            # Existing
├── docker-compose.yml                    # Existing
└── README.md                             # MODIFIED: Updated Azure endpoints
```

---

## Dependencies Summary

### New Dependencies
- **reportlab>=4.1.0**: PDF canvas creation for invisible text layers
- **pypdf>=4.3.0**: PDF manipulation and page merging
- **httpx>=0.27**: Async HTTP client for internal calls (now a direct dependency)

### Dev Dependencies (`[dependency-groups] dev`, excluded from Docker image via `uv sync --no-dev`)
- **pytest>=8.0**: Unit test runner for `tests/test_azure_api.py`

### Existing Dependencies (Used by Azure Layer)
- **fastapi**: API framework
- **paddleocr**: Core OCR engine
- **python-multipart**: Multipart form handling

---

## Troubleshooting

### Issue: "Operation not found" (404)

**Cause**: Task ID not found in TASK_STORAGE

**Solution**:
1. Verify PDF was successfully submitted
2. Check internal PaddleOCR task queue
3. Ensure shared storage is properly initialized

### Issue: "PDF not ready" (404)

**Cause**: Task not completed or PDF generation failed

**Solution**:
1. Poll `/documentintelligence/operations/{task_id}` first
2. Wait for `status: "succeeded"`
3. Check for errors in PaddleOCR task response

### Issue: Text alignment issues in searchable PDF

**Cause**: Coordinate system conversion mismatch

**Solution**:
1. Review `create_searchable_pdf_layer()` coordinate mapping
2. Adjust font size calculation
3. Verify PDF page dimensions match PaddleOCR expectations

### Issue: Docker build fails

**Cause**: Missing dependencies or build arguments

**Solution**:
1. Verify `pyproject.toml` syntax
2. Check Dockerfile COPY paths
3. Ensure model preparation script completes

---

## Next Steps

### Immediate
1. ✅ Create feature branch from `english-translation`
2. ✅ Commit all changes
3. ✅ Push to remote repository
4. ⏳ Monitor GitHub Actions build
5. ⏳ Deploy test container
6. ⏳ Validate API endpoints
7. ⏳ Integrate with Paperless-ngx

### Future Enhancements
1. Add pagination support for large documents
2. Implement async PDF generation for better performance
3. Add support for Azure's full response schema (tables, selection marks)
4. Implement persistent task storage (database-backed)
5. Enforce the `Authorization: Bearer` key on Azure endpoints (currently accepted but ignored)

---

## References

- [Azure Document Intelligence API Documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [ReportLab Documentation](https://www.reportlab.com/docs/reportlab-userguide.pdf)
- [pypdf Documentation](https://pypdf.readthedocs.io/)

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-08-08 | Implementation Team | Initial implementation |
| 1.1 | 2026-08-15 | Implementation Team | SDK wire compatibility (JSON `base64Source`, generic model id, `analyzeResults` PDF route, image support, `stringIndexType`/`error` fidelity), unit test suite, pypdf future-proofing |

---

**Implementation Status**: ✅ ALL PHASES COMPLETED

**Ready for**: Testing and validation on feature branch