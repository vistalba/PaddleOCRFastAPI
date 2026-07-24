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

# 启动时建表（若不存在）
Base.metadata.create_all(bind=engine)
ensure_task_page_organization_columns(engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动 OCR 任务队列 worker（含崩溃恢复）
    await start_workers()
    yield
    # 关闭 worker 协程，再关闭进程池
    await stop_workers()
    _ocr_pool.shutdown(wait=False)
    _ai_pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Paddle OCR API",
    description="基于 Paddle OCR 和 FastAPI 的自用接口",
    lifespan=lifespan,
)

# slowapi 限流
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"resultcode": 429, "message": "请求过于频繁，请稍后再试", "data": []},
    )


# 跨域设置
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
