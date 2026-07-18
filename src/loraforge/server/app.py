"""FastAPI server — thin HTTP/WS translation over the library layer.

Routes contain NO business logic: they call the probe/resolver/downloader/
runner and translate results into HTTP codes and WebSocket frames. Library
pydantic models are reused in responses so the OpenAPI schema stays truthful.

Dependencies arrive via the ``ServerDeps`` dataclass passed to ``create_app``
— tests substitute fakes there; ``run.build_default_deps()`` wires the real
thing. Binding policy (loopback only by default) lives in ``run.py``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import ValidationError

from loraforge.capability.resolver import resolve
from loraforge.recipes.schema import Recipe, validation_messages
from loraforge.server.schemas import (
    DiagnoseResponse,
    DownloadStatus,
    ModelStatus,
    ValidateResponse,
)
from loraforge.server.security import LocalRequestsOnly

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from loraforge.jobs.runner import JobRunner
    from loraforge.probe import HardwareReport
    from loraforge.server.downloads import DownloadManager

_RECIPE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_TERMINAL_JOB_STATES = ("completed", "failed", "cancelled")


@dataclass
class ServerDeps:
    """Everything the routes call. Substitute fakes here in tests."""

    runner: JobRunner
    downloads: DownloadManager
    recipes_dir: Path
    jobs_root: Path
    probe: Callable[[], HardwareReport]
    matrix: dict[str, Any] | None = None  # None → the bundled capability matrix


def _payload(event: Any) -> dict[str, Any]:
    """Dataclass event → JSON-safe dict (Paths and enums become strings)."""
    return json.loads(json.dumps(dataclasses.asdict(event), default=str))


def create_app(deps: ServerDeps, local_only: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        deps.recipes_dir.mkdir(parents=True, exist_ok=True)
        deps.jobs_root.mkdir(parents=True, exist_ok=True)
        yield
        await deps.runner.close()

    app = FastAPI(title="LoRAForge", lifespan=lifespan)
    app.state.deps = deps
    if local_only:  # off only under --allow-remote, where the user fronts their own auth
        app.add_middleware(LocalRequestsOnly)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @app.get("/diagnose", response_model=DiagnoseResponse)
    def diagnose() -> DiagnoseResponse:
        report = deps.probe()
        return DiagnoseResponse(hardware=report, capabilities=resolve(report, deps.matrix))

    # ── Models ───────────────────────────────────────────────────────────────

    @app.get("/models", response_model=list[ModelStatus])
    def models() -> list[ModelStatus]:
        capabilities = resolve(deps.probe(), deps.matrix)
        return [
            ModelStatus(capability=m, download_state=deps.downloads.status(m.model_key))
            for m in capabilities.models
        ]

    @app.post("/models/{model_key}/download", response_model=DownloadStatus, status_code=202)
    async def start_download(model_key: str) -> DownloadStatus:
        if model_key not in deps.downloads.known_models:
            raise HTTPException(404, f"unknown model '{model_key}'")
        started = deps.downloads.start(model_key)
        return DownloadStatus(
            model_key=model_key, state=deps.downloads.status(model_key), started=started
        )

    @app.websocket("/models/{model_key}/events")
    async def download_events(websocket: WebSocket, model_key: str) -> None:
        await websocket.accept()
        if not deps.downloads.has_stream(model_key):
            await websocket.close(code=4004, reason=f"no download started for '{model_key}'")
            return
        try:
            async for event in deps.downloads.events(model_key):
                await websocket.send_json(_payload(event))
        except WebSocketDisconnect:
            return
        await websocket.close()

    # ── Recipes ──────────────────────────────────────────────────────────────

    def recipe_path(name: str) -> Path:
        if not _RECIPE_NAME.match(name):
            raise HTTPException(
                400, "recipe names may only contain letters, digits, '.', '-' and '_'"
            )
        return deps.recipes_dir / f"{name}.yaml"

    @app.get("/recipes", response_model=list[str])
    def list_recipes() -> list[str]:
        return sorted(path.stem for path in deps.recipes_dir.glob("*.yaml"))

    @app.get("/recipes/{name}", response_model=Recipe)
    def get_recipe(name: str) -> Recipe:
        path = recipe_path(name)
        if not path.exists():
            raise HTTPException(404, f"no recipe named '{name}'")
        try:
            return Recipe.from_yaml(path)
        except ValidationError as exc:  # hand-edited file gone bad: still speak human
            raise HTTPException(422, detail=validation_messages(exc)) from exc

    @app.put("/recipes/{name}", response_model=Recipe)
    def put_recipe(name: str, recipe: Recipe) -> Recipe:
        recipe.to_yaml(recipe_path(name))
        return recipe

    @app.delete("/recipes/{name}", status_code=204)
    def delete_recipe(name: str) -> None:
        path = recipe_path(name)
        if not path.exists():
            raise HTTPException(404, f"no recipe named '{name}'")
        path.unlink()

    @app.post("/recipes/validate", response_model=ValidateResponse)
    def validate_recipe(document: dict[str, Any]) -> ValidateResponse:
        try:
            Recipe.model_validate(document)
        except ValidationError as exc:
            return ValidateResponse(valid=False, errors=validation_messages(exc))
        return ValidateResponse(valid=True)

    # ── Jobs ─────────────────────────────────────────────────────────────────

    def job_record(job_id: str) -> dict[str, Any]:
        path = deps.jobs_root / job_id / "job.json"
        if not path.exists():
            raise HTTPException(404, f"no job with id '{job_id}'")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/jobs", status_code=201)
    async def submit_job(recipe: Recipe) -> dict[str, Any]:
        job = await deps.runner.submit(recipe)
        return job_record(job.id)

    @app.get("/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(deps.jobs_root.glob("*/job.json"))
        ]

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return job_record(job_id)

    @app.get("/jobs/{job_id}/artifact")
    def job_artifact(job_id: str) -> FileResponse:
        artifact = job_record(job_id).get("artifact")
        if not artifact or not Path(artifact).exists():
            raise HTTPException(
                404, "no artifact yet — it appears once the job completes successfully"
            )
        return FileResponse(
            artifact, filename=Path(artifact).name, media_type="application/octet-stream"
        )

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            await deps.runner.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return job_record(job_id)

    @app.websocket("/jobs/{job_id}/events")
    async def job_events(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        try:
            deps.runner.get(job_id)
        except KeyError:
            await websocket.close(code=4004, reason=f"no job with id '{job_id}'")
            return
        try:  # replay + live straight from the runner; its stream ends terminally
            async for event in deps.runner.events(job_id):
                await websocket.send_json(_payload(event))
        except WebSocketDisconnect:
            return
        await websocket.close()

    return app
