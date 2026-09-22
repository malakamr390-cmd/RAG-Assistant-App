from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.services.retrieval import load_retrieval_components
from backend.app.api.routes.query import router as query_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts
    load_retrieval_components()
    yield
    # (nothing needed on shutdown)

app = FastAPI(title="RAG Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)