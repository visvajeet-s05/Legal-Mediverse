import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from backend.app.core.config import settings
from backend.app.core.sentry import init_sentry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Initialize Sentry for error tracking (no-op if DSN not configured)
init_sentry()

app = FastAPI(
    title="Legal Mediverse API",
    description="Enterprise-grade multi-agent platform for health, education, community crowdfunding, and legal services.",
    version="1.0.0"
)

# ── Rate Limiting Middleware ───────────────────────────────────────────
from backend.app.core.middleware import add_rate_limiting
add_rate_limiting(app, redis_url=getattr(settings, "REDIS_URL", None))

# ── Prometheus Metrics Endpoint ──────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
logger.info("Prometheus metrics exposed at /metrics")

# Parse allowed origins from environment variable
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
origins = [origin.strip() for origin in allowed_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.edu import router as edu_router
from backend.app.api.v1.community import router as community_router
from backend.app.api.v1.law import router as law_router

app.include_router(auth_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")
app.include_router(edu_router, prefix="/api/v1")
app.include_router(community_router, prefix="/api/v1")
app.include_router(law_router, prefix="/api/v1")

static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "static"))
app.mount(
    "/static",
    StaticFiles(directory=static_dir),
    name="static",
)

from backend.app.core.database import engine, Base
from backend.app.models.models import *


@app.on_event("startup")
async def startup_event():
    # ── Database Initialization ─────────────────────────────────────────
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables successfully initialized on MySQL.")
    except Exception as e:
        logger.warning(f"MySQL connection failed: {e}. Falling back to local SQLite database...")
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
            from backend.app.core import database

            sqlite_engine = create_async_engine("sqlite+aiosqlite:///./mediverse.db", echo=False)
            database.engine = sqlite_engine
            database.AsyncSessionLocal = async_sessionmaker(
                bind=sqlite_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            async with sqlite_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables successfully initialized on local SQLite (mediverse.db).")
        except Exception as sq_err:
            logger.error(f"Failed to initialize local SQLite database: {sq_err}")

    # ── Start Escrow Event Indexer ─────────────────────────────────────
    contract_addr = getattr(settings, "ESCROW_CONTRACT_ADDRESS", "")
    if contract_addr and contract_addr != "0x0000000000000000000000000000000000000000":
        try:
            from backend.app.services.escrow_indexer import EscrowIndexer

            # Determine RPC URL based on environment
            if settings.APP_ENV.value == "production":
                rpc_url = getattr(
                    settings, "POLYGON_MAINNET_RPC_URL",
                    "https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY"
                )
            else:
                rpc_url = getattr(
                    settings, "POLYGON_AMOY_RPC_URL",
                    "https://rpc-amoy.polygon.technology"
                )

            indexer = EscrowIndexer(
                rpc_url=rpc_url,
                contract_address=contract_addr,
                poll_interval=12,
            )
            asyncio.create_task(indexer.start())
            logger.info("Escrow event indexer background task started.")
        except Exception as idx_err:
            logger.warning("Failed to start escrow indexer: %s", idx_err)
    else:
        logger.info("ESCROW_CONTRACT_ADDRESS not set. Escrow indexer not started.")

# Basic REST routes
@app.get("/")
def read_root():
    return {"message": "Welcome to Legal Mediverse API"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "legal-mediverse-api"}

# Active WebSocket connections registry
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket client connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message to client: {e}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Echo or process incoming WebSocket events
            data = await websocket.receive_json()
            logger.info(f"Received WebSocket data: {data}")
            # Broadcast received events back (or process internally)
            await manager.broadcast({"source": "server", "payload": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        manager.disconnect(websocket)
