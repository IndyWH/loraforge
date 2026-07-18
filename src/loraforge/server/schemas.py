"""Response wrappers that compose the library's own pydantic models.

Nothing here re-describes data the library already models — HardwareReport,
CapabilityReport, ModelCapability and Recipe are reused as-is so the OpenAPI
schema stays truthful to what the code actually returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from loraforge.capability.resolver import CapabilityReport, ModelCapability
from loraforge.probe import HardwareReport

DownloadStateName = Literal["not_downloaded", "downloading", "downloaded", "failed"]


class DiagnoseResponse(BaseModel):
    """What the UI's diagnostics page renders."""

    hardware: HardwareReport
    capabilities: CapabilityReport


class ModelStatus(BaseModel):
    """Capability verdict merged with local download state and source facts."""

    capability: ModelCapability
    download_state: DownloadStateName
    download_gb: float | None = None
    gated: bool | None = None  # needs the HF licence-acceptance step


class DownloadStatus(BaseModel):
    model_key: str
    state: DownloadStateName
    started: bool  # False when already downloading/downloaded (idempotent POST)


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)  # human-worded, verbatim


class DatasetCreate(BaseModel):
    name: str


class DatasetCreated(BaseModel):
    name: str
    path: Path  # what a recipe's dataset.path should reference


class IngestRequest(BaseModel):
    sources: list[Path]  # local files chosen by the user; copied, never moved


class CaptionPayload(BaseModel):
    caption: str


class CaptionResponse(BaseModel):
    filename: str
    caption: str | None  # None: image exists but has no sidecar yet


class TriggerRequest(BaseModel):
    trigger_word: str


class TriggerResponse(BaseModel):
    updated: int  # captions created or changed (idempotent: 0 on re-run)
