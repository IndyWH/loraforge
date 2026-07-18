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
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from loraforge.capability.resolver import resolve
from loraforge.datasets.library import DatasetSummary, IngestResult
from loraforge.recipes.schema import Recipe, validation_messages
from loraforge.server.schemas import (
    CaptionPayload,
    CaptionResponse,
    DatasetCreate,
    DatasetCreated,
    DiagnoseResponse,
    DownloadStatus,
    IngestRequest,
    ModelStatus,
    TriggerRequest,
    TriggerResponse,
    ValidateResponse,
)
from loraforge.server.security import LocalRequestsOnly

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from loraforge.datasets.library import DatasetLibrary
    from loraforge.jobs.runner import JobRunner
    from loraforge.probe import HardwareReport
    from loraforge.server.downloads import DownloadManager

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_TERMINAL_JOB_STATES = ("completed", "failed", "cancelled")


@dataclass
class ServerDeps:
    """Everything the routes call. Substitute fakes here in tests."""

    runner: JobRunner
    downloads: DownloadManager
    datasets: DatasetLibrary
    recipes_dir: Path
    jobs_root: Path
    probe: Callable[[], HardwareReport]
    matrix: dict[str, Any] | None = None  # None → the bundled capability matrix
    ui_dist: Path | None = None  # built web UI to serve at / (ui/dist), if present


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
    # CORS headers for the Vite dev server (5173) and other loopback origins;
    # foreign origins get no CORS grant (and LocalRequestsOnly 403s their writes).
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @app.get("/diagnose", response_model=DiagnoseResponse)
    def diagnose() -> DiagnoseResponse:
        report = deps.probe()
        return DiagnoseResponse(hardware=report, capabilities=resolve(report, deps.matrix))

    # ── Models ───────────────────────────────────────────────────────────────

    @app.get("/models", response_model=list[ModelStatus])
    def models() -> list[ModelStatus]:
        capabilities = resolve(deps.probe(), deps.matrix)
        entries = []
        for m in capabilities.models:
            facts = deps.downloads.source_facts(m.model_key)
            entries.append(
                ModelStatus(
                    capability=m,
                    download_state=deps.downloads.status(m.model_key),
                    download_gb=facts[0] if facts else None,
                    gated=facts[1] if facts else None,
                )
            )
        return entries

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
        if not _SAFE_NAME.match(name):
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

    # ── Datasets ─────────────────────────────────────────────────────────────

    def dataset_name(name: str) -> str:
        if not _SAFE_NAME.match(name):
            raise HTTPException(
                400, "dataset names may only contain letters, digits, '.', '-' and '_'"
            )
        return name

    @app.get("/datasets", response_model=list[str])
    def list_datasets() -> list[str]:
        return deps.datasets.list_names()

    @app.post("/datasets", response_model=DatasetCreated, status_code=201)
    def create_dataset(body: DatasetCreate) -> DatasetCreated:
        path = deps.datasets.create(dataset_name(body.name))
        return DatasetCreated(name=body.name, path=path)

    @app.get("/datasets/{name}", response_model=DatasetSummary)
    def dataset_status(name: str) -> DatasetSummary:
        try:
            return deps.datasets.status(dataset_name(name))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/datasets/{name}", status_code=204)
    def delete_dataset(name: str) -> None:
        try:
            deps.datasets.delete(dataset_name(name))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/datasets/{name}/images", response_model=IngestResult)
    def ingest_images(name: str, body: IngestRequest) -> IngestResult:
        try:
            return deps.datasets.ingest(dataset_name(name), body.sources)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/datasets/{name}/upload", response_model=IngestResult)
    async def upload_images(name: str, files: list[UploadFile]) -> IngestResult:
        """Browser uploads: spool multipart files to disk, then the exact same
        ingest pipeline (and verdicts) as path-based ingestion."""
        dataset_name(name)
        with tempfile.TemporaryDirectory(prefix="loraforge-upload-") as spool:
            staged: list[Path] = []
            for upload in files:
                safe_name = Path(upload.filename or "unnamed").name  # strip any path parts
                target = Path(spool) / safe_name
                with target.open("wb") as sink:
                    shutil.copyfileobj(upload.file, sink)
                staged.append(target)
            try:
                return deps.datasets.ingest(name, staged)
            except FileNotFoundError as exc:
                raise HTTPException(404, str(exc)) from exc

    @app.get("/datasets/{name}/captions/{filename}", response_model=CaptionResponse)
    def get_caption(name: str, filename: str) -> CaptionResponse:
        try:
            caption = deps.datasets.get_caption(dataset_name(name), filename)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return CaptionResponse(filename=filename, caption=caption)

    @app.put("/datasets/{name}/captions/{filename}", response_model=CaptionResponse)
    def put_caption(name: str, filename: str, body: CaptionPayload) -> CaptionResponse:
        try:
            deps.datasets.set_caption(dataset_name(name), filename, body.caption)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return CaptionResponse(filename=filename, caption=body.caption.strip())

    @app.post("/datasets/{name}/trigger-word", response_model=TriggerResponse)
    def inject_trigger(name: str, body: TriggerRequest) -> TriggerResponse:
        try:
            updated = deps.datasets.inject_trigger_word(dataset_name(name), body.trigger_word)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return TriggerResponse(updated=updated)

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
    async def cancel_job(job_id: str, keep: bool = False) -> dict[str, Any]:
        try:
            await deps.runner.cancel(job_id, keep=keep)
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

    # ── Web UI (built bundle) ────────────────────────────────────────────────
    # Mounted last: API routes above always win; everything else is the app.
    if deps.ui_dist is not None and deps.ui_dist.is_dir():
        app.mount("/", StaticFiles(directory=deps.ui_dist, html=True), name="ui")

    return app
