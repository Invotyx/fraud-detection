"""
FastAPI Application — Phase 13
Main API entrypoint wiring the full pipeline as HTTP endpoints.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from integration.api.config import get_settings
from integration.api.pipeline import run_pipeline
from integration.api.schemas import AnalyzeRequest, AnalyzeResponse
from integration.hitl.queue import (
    escalate_stale,
    get_item,
    get_pending,
    submit_decision,
)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url=None,
    )

    # CORS — restrict in production
    origins = (
        [o.strip() for o in settings.cors_allowed_origins.split(",")]
        if settings.app_env != "production"
        else []
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------------
    # Database lifecycle
    # ----------------------------------------------------------------

    _engine: AsyncEngine | None = None

    @app.on_event("startup")
    async def _startup() -> None:
        nonlocal _engine
        _engine = create_async_engine(
            settings.database_url, pool_pre_ping=True)
        app.state.engine = _engine

        # Initialise pgvector tables and optionally seed fraud patterns
        try:
            from integration.vector_store.store import init_vector_tables
            from integration.vector_store.fraud_patterns import (
                is_knowledge_base_seeded,
                seed_knowledge_base,
            )
            from integration.vector_store.encoder import get_encoder
            import yaml
            import os

            rag_cfg_path = os.path.join(
                os.path.dirname(__file__), "..", "configs", "classifiers.yaml"
            )
            with open(rag_cfg_path) as fh:
                rag_cfg = yaml.safe_load(fh).get("rag", {})
            seed_on_startup: bool = rag_cfg.get("seed_on_startup", True)

            async with _engine.connect() as conn:
                raw = await conn.get_raw_connection()
                driver_conn = raw.driver_connection
                await init_vector_tables(driver_conn)
                if seed_on_startup:
                    already_seeded = await is_knowledge_base_seeded(driver_conn)
                    if not already_seeded:
                        encoder = await asyncio.to_thread(get_encoder)
                        if encoder is not None:
                            count = await seed_knowledge_base(driver_conn, encoder)
                            import logging
                            logging.getLogger(__name__).info(
                                "Fraud pattern knowledge base seeded with %d patterns.", count
                            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Vector store initialisation skipped: %s", exc
            )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if _engine:
            await _engine.dispose()

    # ----------------------------------------------------------------
    # Dependencies
    # ----------------------------------------------------------------

    async def get_db() -> AsyncGenerator[AsyncConnection, None]:
        engine: AsyncEngine = app.state.engine
        async with engine.connect() as conn:
            async with conn.begin():
                yield conn

    def verify_api_key(
        x_api_key: str = Header(alias=settings.api_key_header, default=None),
    ) -> str:
        # In production replace with a proper secrets store lookup
        valid_keys: set[str] = _load_api_keys()
        if not x_api_key or x_api_key not in valid_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )
        return x_api_key

    def verify_hitl_api_key(
        x_api_key: str = Header(alias=settings.api_key_header, default=None),
    ) -> str:
        """Stricter HITL key check — same mechanism, separate dependency for clarity."""
        return verify_api_key(x_api_key)

    # ----------------------------------------------------------------
    # Rate limiting (in-memory token bucket; use Redis in production)
    # ----------------------------------------------------------------

    _rate_buckets: dict[str, list[float]] = {}

    async def rate_limit(
        request: Request,
        api_key: str = Depends(verify_api_key),
    ) -> None:
        now = time.monotonic()
        window = settings.rate_limit_window_seconds
        bucket = _rate_buckets.setdefault(api_key, [])
        # Evict old timestamps
        _rate_buckets[api_key] = [t for t in bucket if now - t < window]
        if len(_rate_buckets[api_key]) >= settings.rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        _rate_buckets[api_key].append(now)

    # ----------------------------------------------------------------
    # Routes
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Dev test UI  (disabled in production)
    # ----------------------------------------------------------------
    if settings.app_env != "production":
        import os as _os
        _DEV_HTML = _os.path.join(
            _os.path.dirname(__file__), "..", "dev_test.html"
        )

        @app.get("/dev", response_class=HTMLResponse, include_in_schema=False)
        async def dev_ui() -> HTMLResponse:
            """Serve the dev test UI. Available only outside production."""
            try:
                with open(_DEV_HTML, "r", encoding="utf-8") as fh:
                    return HTMLResponse(content=fh.read())
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404,
                    detail="dev_test.html not found — run from the integration/ directory.",
                )

    @app.get("/health", tags=["ops"])
    async def health(db: AsyncConnection = Depends(get_db)) -> JSONResponse:
        """Liveness + readiness probe — checks DB connectivity."""
        try:
            from sqlalchemy import text
            await db.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False

        code = 200 if db_ok else 503
        return JSONResponse(
            status_code=code,
            content={"status": "ok" if db_ok else "degraded", "db": db_ok},
        )

    @app.post(
        "/analyze",
        response_model=AnalyzeResponse,
        status_code=status.HTTP_200_OK,
        tags=["pipeline"],
        dependencies=[Depends(rate_limit)],
    )
    async def analyze(
        body: AnalyzeRequest,
        x_trace_id: str = Header(default=None, alias="X-Trace-ID"),
        db: AsyncConnection = Depends(get_db),
    ) -> AnalyzeResponse:
        """
        Full fraud detection pipeline.
        Clients may supply an ``X-Trace-ID`` header; one is generated otherwise.
        """
        trace_id: UUID = _parse_trace_id(x_trace_id)
        response = await run_pipeline(body, db, redis_client=None, trace_id=trace_id)
        return response

    @app.get(
        "/hitl/queue",
        tags=["hitl"],
        dependencies=[Depends(verify_hitl_api_key)],
    )
    async def hitl_list(
        limit: int = settings.hitl_queue_default_limit,
        db: AsyncConnection = Depends(get_db),
    ) -> JSONResponse:
        """List pending HITL review items (auth required)."""
        items = await get_pending(db, limit=limit)
        return JSONResponse(content={"items": items, "count": len(items)})

    @app.get(
        "/hitl/{item_id}",
        tags=["hitl"],
        dependencies=[Depends(verify_hitl_api_key)],
    )
    async def hitl_get(
        item_id: str,
        db: AsyncConnection = Depends(get_db),
    ) -> JSONResponse:
        """Retrieve full detail for a single HITL item (auth required)."""
        item = await get_item(db, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        return JSONResponse(content=item)

    @app.post(
        "/hitl/{item_id}/decision",
        tags=["hitl"],
        dependencies=[Depends(verify_hitl_api_key)],
    )
    async def hitl_decide(
        item_id: str,
        body: dict,
        x_api_key: str = Depends(verify_hitl_api_key),
        db: AsyncConnection = Depends(get_db),
    ) -> JSONResponse:
        """
        Submit a human review decision.
        Body: ``{"decision": "allow"|"block", "notes": "..."}``
        """
        decision = body.get("decision")
        notes = body.get("notes", "")
        if decision not in ("allow", "block"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="decision must be 'allow' or 'block'",
            )
        try:
            await submit_decision(db, item_id, x_api_key, decision, notes)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(content={"status": "ok", "item_id": item_id, "decision": decision})

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_trace_id(raw: str | None) -> UUID:
    if raw:
        try:
            return UUID(raw)
        except ValueError:
            pass
    return uuid4()


def _load_api_keys() -> set[str]:
    """
    Load valid API keys from environment / secrets.
    Extend this to integrate with a proper secrets manager.
    """
    import os
    raw = os.environ.get("API_KEYS", "dev-key-change-me")
    return {k.strip() for k in raw.split(",") if k.strip()}


# ---------------------------------------------------------------------------
# ASGI entrypoint
# ---------------------------------------------------------------------------

app = create_app()
