"""Response wrappers that compose the library's own pydantic models.

Nothing here re-describes data the library already models — HardwareReport,
CapabilityReport, ModelCapability and Recipe are reused as-is so the OpenAPI
schema stays truthful to what the code actually returns.
"""

from __future__ import annotations

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
    """Capability verdict merged with local download state."""

    capability: ModelCapability
    download_state: DownloadStateName


class DownloadStatus(BaseModel):
    model_key: str
    state: DownloadStateName
    started: bool  # False when already downloading/downloaded (idempotent POST)


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)  # human-worded, verbatim
