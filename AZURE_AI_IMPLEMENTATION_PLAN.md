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

#### Test 1: Submit PDF for Analysis

```bash
curl -X POST "http://localhost:8000/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test-document.pdf"
```

**Expected Response**: `202 Accepted`
```
Operation-Location: http://localhost:8000/documentintelligence/operations/{task_id}?api-version=2024-11-30
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
    "modelId": "prebuilt-layout",
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
├── pyproject.toml                        # MODIFIED: Added dependencies
├── Dockerfile                            # Existing
├── docker-compose.yml                    # Existing
└── README.md                             # Existing
```

---

## Dependencies Summary

### New Dependencies
- **reportlab>=4.1.0**: PDF canvas creation for invisible text layers
- **pypdf>=4.3.0**: PDF manipulation and page merging

### Existing Dependencies (Used by Azure Layer)
- **fastapi**: API framework
- **httpx**: Async HTTP client for internal calls
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
5. Add rate limiting for Azure-compatible endpoints
6. Support for custom model IDs

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

---

**Implementation Status**: ✅ ALL PHASES COMPLETED

**Ready for**: Testing and validation on feature branch