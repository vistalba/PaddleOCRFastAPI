# -*- coding: utf-8 -*-

from contextlib import asynccontextmanager

# import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from database import Base, engine
from limiter import limiter
from models.RestfulModel import *
from models import TaskModel  # noqa: F401 – ensure table is registered before create_all
from routers import ocr
from routers import tasks
from routers.tasks import _ai_pool, _ocr_pool, start_workers, stop_workers
from schema_migrations import ensure_task_page_organization_columns
from utils.ImageHelper import *

# Create tables at startup (if they don't exist)
Base.metadata.create_all(bind=engine)
ensure_task_page_organization_columns(engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start OCR task queue workers (with crash recovery)
    await start_workers()
    yield
    # Close worker coroutines, then shutdown process pool
    await stop_workers()
    _ocr_pool.shutdown(wait=False)
    _ai_pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Paddle OCR API",
    description="Personal OCR API based on PaddleOCR and FastAPI",
    lifespan=lifespan,
)

# slowapi rate limiting
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"resultcode": 429, "message": "Too many requests, please try again later", "data": []},
    )


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr.router)
app.include_router(tasks.router)

# uvicorn.run(app=app, host="0.0.0.0", port=48301)
