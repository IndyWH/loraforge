"""Async job runner — one GPU, one job at a time, OOM survived.

Transport-free by design: no FastAPI or WebSocket imports here. The runner
exposes ``submit()`` / ``cancel()`` / ``events()`` — an async event stream
per job — and the server layer (roadmap: FastAPI) wraps those in routes and
sockets later. Jobs queue FIFO; a single worker drains the queue because a
single consumer GPU can only train one thing at a time.

Lifecycle is an explicit state machine::

    queued → preparing → running → completed | failed | cancelled
                 ↑           │
                 └── oom_stepdown  (≤ max_retries loops)

Every state change is a JobEvent; every event stream ends with a terminal
event. A job record (recipe snapshot, state history, step-downs, log path,
artifact path) is persisted as ``job.json`` in the job's workdir after every
transition — crash-safe, and exactly what the server and UI will read. The
recorded step-downs are real-world VRAM data the capability matrix can be
tightened with later.

Cancellation is portable: Windows has no SIGTERM, so process shutdown goes
through psutil over the whole process tree (accelerate spawns children) —
terminate first, escalate to kill after a grace period.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import psutil

from loraforge.jobs.stepdown import step_down, vram_knobs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from loraforge.engines.base import EngineAdapter, LaunchPlan, ProgressEvent
    from loraforge.recipes.schema import Recipe

# ── States and events ────────────────────────────────────────────────────────


class JobState(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"  # compile recipe, write config_files
    RUNNING = "running"
    OOM_STEPDOWN = "oom_stepdown"  # transient: tighter recipe chosen, loops to preparing
    COMPLETED = "completed"
    COMPLETED_EARLY = "completed_early"  # stop-and-keep: user stopped, newest checkpoint kept
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset(
    {JobState.COMPLETED, JobState.COMPLETED_EARLY, JobState.FAILED, JobState.CANCELLED}
)


@dataclass(frozen=True)
class JobEvent:
    """One item in a job's event stream: a state change or a progress line."""

    job_id: str
    seq: int
    kind: Literal["state", "progress"]
    state: JobState  # the job's state when the event was emitted
    progress: ProgressEvent | None = None  # set when kind == "progress"
    message: str | None = None  # human words, when there is something to say

    @property
    def is_terminal(self) -> bool:
        return self.kind == "state" and self.state in TERMINAL_STATES


@dataclass
class Job:
    id: str
    recipe: Recipe  # current recipe (tightened by step-downs)
    workdir: Path
    state: JobState = JobState.QUEUED
    state_history: list[dict[str, Any]] = field(default_factory=list)
    stepdowns: list[dict[str, Any]] = field(default_factory=list)
    artifact: Path | None = None
    error: str | None = None
    cancel_requested: bool = False
    keep_requested: bool = False  # stop-and-keep: collect the newest checkpoint on stop
    history: list[JobEvent] = field(default_factory=list, repr=False)
    subscribers: list[asyncio.Queue[JobEvent]] = field(default_factory=list, repr=False)
    proc: Any = field(default=None, repr=False)  # live process handle while running

    @property
    def log_path(self) -> Path:
        return self.workdir / "job.log"

    @property
    def record_path(self) -> Path:
        return self.workdir / "job.json"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ── Subprocess handling (portable, tree-wide) ────────────────────────────────


def _signal_tree(pid: int, *, hard: bool) -> None:
    """Terminate/kill a process and all its children (accelerate spawns some)."""
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for proc in [*root.children(recursive=True), root]:
        with contextlib.suppress(psutil.NoSuchProcess):
            proc.kill() if hard else proc.terminate()


class _EngineProcess:
    """Default process handle; tests substitute a fake with the same surface."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    async def next_line(self) -> str | None:
        assert self._proc.stdout is not None
        raw = await self._proc.stdout.readline()
        return raw.decode("utf-8", errors="replace") if raw else None

    async def wait(self) -> int:
        return await self._proc.wait()

    def terminate(self) -> None:
        _signal_tree(self._proc.pid, hard=False)

    def kill(self) -> None:
        _signal_tree(self._proc.pid, hard=True)


async def _spawn_subprocess(plan: LaunchPlan) -> _EngineProcess:
    proc = await asyncio.create_subprocess_exec(
        *plan.argv,
        cwd=plan.cwd,
        env={**os.environ, **plan.env},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,  # kohya logs tqdm to stderr; one stream
    )
    return _EngineProcess(proc)


# ── The runner ───────────────────────────────────────────────────────────────

_OOM_ADVICE = (
    "Close other applications using the GPU (browsers, games, image viewers), "
    "or pick a smaller model, then start the job again."
)


class JobRunner:
    """FIFO queue + single worker over an EngineAdapter. See module docstring."""

    def __init__(
        self,
        adapter: EngineAdapter,
        jobs_root: Path,
        spawn: Callable[[LaunchPlan], Awaitable[Any]] | None = None,
        max_retries: int = 2,  # OOM step-down retries per job
        grace_seconds: float = 10.0,  # terminate → kill escalation window
    ) -> None:
        self.adapter = adapter
        self.jobs_root = jobs_root
        self.max_retries = max_retries
        self.grace_seconds = grace_seconds
        self._spawn = spawn or _spawn_subprocess
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    # ── Public API (what the FastAPI layer will wrap) ────────────────────────

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"no job with id '{job_id}'") from None

    async def submit(self, recipe: Recipe) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, recipe=recipe, workdir=self.jobs_root / job_id)
        job.workdir.mkdir(parents=True, exist_ok=True)
        self._jobs[job.id] = job
        busy = self._queue.qsize() > 0 or any(
            j.state in (JobState.PREPARING, JobState.RUNNING, JobState.OOM_STEPDOWN)
            for j in self._jobs.values()
        )
        self._transition(
            job,
            JobState.QUEUED,
            "Waiting for the GPU — one job runs at a time." if busy else None,
        )
        self._queue.put_nowait(job)
        if self._worker is None:
            self._worker = asyncio.create_task(self._work())
        return job

    async def cancel(self, job_id: str, keep: bool = False) -> None:
        """Stop a job. With ``keep=True``, the newest saved checkpoint is
        collected and the job finishes as ``completed_early`` instead of
        ``cancelled`` (falls back to cancelled if nothing was saved yet)."""
        job = self.get(job_id)
        if job.state in TERMINAL_STATES:
            return
        job.keep_requested = keep
        job.cancel_requested = True
        if job.state is JobState.QUEUED:  # nothing ran, so nothing to keep
            self._transition(job, JobState.CANCELLED, "Cancelled before training started.")
            return
        if job.proc is not None:  # unblock the worker's read loop
            await self._stop(job.proc)

    async def cancel_all(self, keep: bool = False) -> list[str]:
        """Cancel every queued/running job (the shutdown path); returns their ids.

        Each cancellation goes through :meth:`cancel`, so a mid-run job still
        gets the checkpointed stop and its terminal event — which also lets
        any attached event stream end instead of holding shutdown open.
        """
        cancelled = []
        for job_id, job in list(self._jobs.items()):
            if job.state not in TERMINAL_STATES:
                await self.cancel(job_id, keep=keep)
                cancelled.append(job_id)
        return cancelled

    async def events(self, job_id: str) -> AsyncIterator[JobEvent]:
        """Replay a job's full event history, then follow live events.

        Always ends with (and only after) a terminal event, so consumers can
        simply `async for` until the stream closes.
        """
        job = self.get(job_id)
        queue: asyncio.Queue[JobEvent] = asyncio.Queue()
        job.subscribers.append(queue)  # subscribe first, dedupe by seq below
        try:
            last_seq = -1
            for event in list(job.history):
                yield event
                last_seq = event.seq
                if event.is_terminal:
                    return
            while True:
                event = await queue.get()
                if event.seq <= last_seq:
                    continue
                yield event
                if event.is_terminal:
                    return
        finally:
            job.subscribers.remove(queue)

    async def close(self) -> None:
        """Shut down the worker; a mid-run job is cancelled with a terminal event."""
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    # ── Worker ───────────────────────────────────────────────────────────────

    async def _work(self) -> None:
        while True:
            job = await self._queue.get()
            if job.state in TERMINAL_STATES:  # cancelled while queued
                continue
            try:
                await self._execute(job)
            except asyncio.CancelledError:
                if job.state not in TERMINAL_STATES:
                    self._transition(job, JobState.CANCELLED, "Runner shut down.")
                raise
            except Exception as exc:  # the worker must survive any single job
                self._transition(job, JobState.FAILED, f"Unexpected error: {exc}")

    async def _execute(self, job: Job) -> None:
        retries = 0
        while True:
            if job.cancel_requested:
                self._transition(job, JobState.CANCELLED, "Training cancelled.")
                return

            self._transition(job, JobState.PREPARING)
            try:
                plan = self.adapter.compile(job.recipe, job.workdir)
            except (ValueError, FileNotFoundError) as exc:
                self._transition(job, JobState.FAILED, str(exc))
                return
            for path, content in plan.config_files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            self._transition(job, JobState.RUNNING, f"Training started (attempt {retries + 1}).")
            proc = await self._spawn(plan)
            job.proc = proc
            oom: ProgressEvent | None = None
            with job.log_path.open("a", encoding="utf-8") as log:
                while not job.cancel_requested:
                    line = await proc.next_line()
                    if line is None:
                        break
                    log.write(line if line.endswith("\n") else line + "\n")
                    event = self.adapter.parse_line(line)
                    if event is None:
                        continue
                    self._emit(job, "progress", progress=event, message=event.message)
                    if event.is_oom:
                        oom = event
                        break
            job.proc = None

            if job.cancel_requested:
                await self._stop(proc)
                if job.keep_requested:
                    try:
                        result = self.adapter.collect(job.workdir)
                    except FileNotFoundError:
                        self._transition(
                            job,
                            JobState.CANCELLED,
                            "Training stopped — no checkpoint had been saved yet, "
                            "so there was nothing to keep.",
                        )
                        return
                    job.artifact = result.artifact
                    self._transition(
                        job,
                        JobState.COMPLETED_EARLY,
                        f"Training stopped early — kept the latest saved LoRA: {result.artifact}",
                    )
                    return
                self._transition(job, JobState.CANCELLED, "Training cancelled.")
                return

            if oom is not None:
                await self._stop(proc)
                tried = [s["message"] for s in job.stepdowns]
                if retries >= self.max_retries:
                    self._transition(
                        job,
                        JobState.FAILED,
                        "Your GPU ran out of memory even after automatic adjustments "
                        f"(tried: {' then: '.join(tried)}). {_OOM_ADVICE}",
                    )
                    return
                stepped = step_down(job.recipe)
                if stepped is None:
                    already = f" Already tried: {' then: '.join(tried)}." if tried else ""
                    self._transition(
                        job,
                        JobState.FAILED,
                        "Your GPU ran out of memory and there are no tighter settings "
                        f"left to try.{already} {_OOM_ADVICE}",
                    )
                    return
                job.stepdowns.append(
                    {
                        "at": _now(),
                        "message": stepped.message,
                        "from": vram_knobs(job.recipe),
                        "to": vram_knobs(stepped.recipe),
                    }
                )
                job.recipe = stepped.recipe
                retries += 1
                self._transition(
                    job,
                    JobState.OOM_STEPDOWN,
                    f"Your GPU ran out of memory. {stepped.message} "
                    f"Retrying ({retries}/{self.max_retries}).",
                )
                continue

            returncode = await proc.wait()
            if returncode != 0:
                self._transition(
                    job,
                    JobState.FAILED,
                    f"Training exited with code {returncode} — full log: {job.log_path}",
                )
                return
            try:
                result = self.adapter.collect(job.workdir)
            except FileNotFoundError as exc:
                self._transition(job, JobState.FAILED, str(exc))
                return
            job.artifact = result.artifact
            self._transition(
                job, JobState.COMPLETED, f"Training complete — LoRA saved to {result.artifact}"
            )
            return

    async def _stop(self, proc: Any) -> None:
        """Graceful tree shutdown, escalating to kill after the grace period."""
        if proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.grace_seconds)
        except TimeoutError:
            proc.kill()
            await proc.wait()

    # ── Events + persistence ─────────────────────────────────────────────────

    def _emit(
        self,
        job: Job,
        kind: Literal["state", "progress"],
        progress: ProgressEvent | None = None,
        message: str | None = None,
    ) -> None:
        event = JobEvent(
            job_id=job.id,
            seq=len(job.history),
            kind=kind,
            state=job.state,
            progress=progress,
            message=message,
        )
        job.history.append(event)
        for queue in job.subscribers:
            queue.put_nowait(event)

    def _transition(self, job: Job, state: JobState, message: str | None = None) -> None:
        job.state = state
        if state is JobState.FAILED:
            job.error = message
        job.state_history.append({"state": state.value, "at": _now(), "message": message})
        self._emit(job, "state", message=message)
        self._persist(job)

    def _persist(self, job: Job) -> None:
        record = {
            "id": job.id,
            "name": job.recipe.name,
            "model": job.recipe.model,
            "state": job.state.value,
            "recipe": job.recipe.model_dump(mode="json"),
            "state_history": job.state_history,
            "stepdowns": job.stepdowns,  # real-world VRAM data → capability matrix
            "log_path": str(job.log_path),
            "artifact": str(job.artifact) if job.artifact else None,
            "error": job.error,
        }
        job.record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
