import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_logger, setup_logging
from app.routers import task_router

load_dotenv()
setup_logging()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Resource API starting up...")
    yield
    # Shutdown
    logger.info("Resource API shutting down...")


app = FastAPI(
    title="Resource API",
    description="API for managing in-memory resource X",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
allow_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(task_router.router, prefix="/api", tags=["task"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
