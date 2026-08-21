from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from interview_coach import __version__
from interview_coach.api.routes import build_router
from interview_coach.config import Settings
from interview_coach.container import create_container
from interview_coach.exceptions import InterviewCoachError, SessionNotFoundError
from interview_coach.logging_config import configure_logging


def create_app(
    settings: Settings | None = None,
    *,
    serve_frontend: bool = True,
) -> FastAPI:
    container = create_container(settings)
    configure_logging(container.settings.log_level)
    app = FastAPI(
        title="SignalPrep API",
        version=__version__,
        description=(
            "Grounded interview-practice and evaluation software; never an automated hiring system."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.container = container
    app.include_router(build_router(container.orchestrator))
    # Vercel maps api/index.py to /api/* and preserves the full request path.
    # Keeping both prefixes also makes the compiled frontend portable to Uvicorn.
    app.include_router(build_router(container.orchestrator), prefix="/api")

    @app.exception_handler(InterviewCoachError)
    async def controlled_error(request: Request, exc: InterviewCoachError):
        status = 404 if isinstance(exc, SessionNotFoundError) else 422
        return JSONResponse(
            status_code=status, content={"detail": str(exc), "type": type(exc).__name__}
        )

    @app.exception_handler(ValidationError)
    async def validation_error(request: Request, exc: ValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors(include_url=False)})

    candidates = (
        Path(__file__).resolve().parents[1] / "frontend_dist",
        Path(__file__).resolve().parents[3] / "frontend" / "dist",
    )
    frontend_dist = next((path for path in candidates if path.is_dir()), candidates[-1])
    if serve_frontend and frontend_dist.is_dir():
        assets = frontend_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        async def frontend_index():
            return FileResponse(frontend_dist / "index.html")

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("interview_coach.api.app:app", host="127.0.0.1", port=8000, reload=False)
