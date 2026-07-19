"""Transport glue: bridge the sync downloader into async event streams.

No download logic lives here — the ModelDownloader decides what to fetch and
emits the events; this manager only runs it in a thread, collects the events
on the loop, and replays them to any number of async subscribers with the
same contract as the job runner's streams (replay + live, terminal event
guaranteed).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loraforge.downloader import DownloadError, DownloadEvent, DownloadState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from loraforge.downloader import DownloadedModel, ModelDownloader
    from loraforge.server.schemas import DownloadStateName


class DownloadManager:
    def __init__(
        self,
        downloader: ModelDownloader,
        on_complete: Callable[[DownloadedModel], None] | None = None,
    ) -> None:
        self._downloader = downloader
        self._on_complete = on_complete  # e.g. wire new paths into the engine adapter
        self._history: dict[str, list[DownloadEvent]] = {}
        self._wakeups: dict[str, list[asyncio.Queue[None]]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._results: dict[str, DownloadedModel] = {}

    @property
    def known_models(self) -> set[str]:
        return set(self._downloader.sources)

    def source_facts(self, model_key: str) -> tuple[float, bool] | None:
        """(download_gb, gated) for the UI's model cards; None if unknown."""
        source = self._downloader.sources.get(model_key)
        return None if source is None else (source.download_gb, source.gated)

    def status(self, model_key: str) -> DownloadStateName:
        if model_key in self._results:
            return "downloaded"
        # A terminal failed event outranks the task: stop_all() pushes one
        # before the cancelled task has had a chance to finish.
        history = self._history.get(model_key)
        if history and history[-1].state is DownloadState.FAILED:
            return "failed"
        task = self._tasks.get(model_key)
        if task is not None and not task.done():
            return "downloading"
        if self._downloader.peek(model_key) is not None:
            return "downloaded"  # cached before this session (or by another tool)
        return "not_downloaded"

    def has_stream(self, model_key: str) -> bool:
        return model_key in self._history

    def start(self, model_key: str) -> bool:
        """Kick off a download task; no-op (False) if underway or already local."""
        if self.status(model_key) in ("downloading", "downloaded"):
            return False
        self._history[model_key] = []
        loop = asyncio.get_running_loop()

        def on_event(event: DownloadEvent) -> None:  # called from the worker thread
            loop.call_soon_threadsafe(self._push, model_key, event)

        async def run() -> None:
            try:
                result = await asyncio.to_thread(self._downloader.download, model_key, on_event)
            except DownloadError:
                return  # the downloader already emitted the terminal failed event
            self._results[model_key] = result
            if self._on_complete is not None:
                self._on_complete(result)

        self._tasks[model_key] = asyncio.create_task(run())
        return True

    def stop_all(self) -> None:
        """Abandon in-flight downloads (the shutdown path).

        The hub call runs in a thread that can't be interrupted mid-file, so
        this cancels the wrapping task (no completion wiring will fire) and
        pushes a terminal event so subscribed streams end instead of holding
        shutdown open. Partial files resume from the HF cache next launch.
        """
        for model_key, task in self._tasks.items():
            if task.done():
                continue
            task.cancel()
            self._push(
                model_key,
                DownloadEvent(
                    model_key=model_key,
                    state=DownloadState.FAILED,
                    message="Download stopped — LoRAForge is shutting down. "
                    "It will pick up where it left off next time.",
                ),
            )

    async def events(self, model_key: str) -> AsyncIterator[DownloadEvent]:
        """Replay this session's events for a download, then follow live ones."""
        queue: asyncio.Queue[None] = asyncio.Queue()
        self._wakeups.setdefault(model_key, []).append(queue)
        try:
            index = 0
            while True:
                history = self._history.get(model_key, [])
                while index < len(history):
                    event = history[index]
                    index += 1
                    yield event
                    if event.is_terminal:
                        return
                await queue.get()  # wakeup signal; the loop above drains new history
        finally:
            self._wakeups[model_key].remove(queue)

    def _push(self, model_key: str, event: DownloadEvent) -> None:
        self._history[model_key].append(event)
        for queue in self._wakeups.get(model_key, []):
            queue.put_nowait(None)
