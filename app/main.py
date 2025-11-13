import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import agent_router

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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

allow_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent_router.router, prefix="/api", tags=["agent"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
