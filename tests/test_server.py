"""Server tests: thin routes over injected fakes, WS streams, loopback guard.

Everything runs through the ASGI test client — no sockets, no GPU, no
network. The runner gets the fake adapter/process from the job runner tests;
the downloader gets the fake hub from the downloader tests.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_capability import fake_report
from test_datasets import fake_photo
from test_downloader import make_downloader
from test_job_runner import FakeAdapter, FakeProcess, FakeSpawner, make_recipe

from loraforge.datasets.library import DatasetLibrary
from loraforge.jobs.runner import JobRunner
from loraforge.server.app import ServerDeps, create_app
from loraforge.server.downloads import DownloadManager
from loraforge.server.run import ensure_local_bind

RTX_3060 = fake_report("NVIDIA GeForce RTX 3060", 12288, 11400, (8, 6))


def make_client(tmp_path, procs=()) -> tuple[TestClient, ServerDeps]:
    downloader, _hub = make_downloader(tmp_path)
    deps = ServerDeps(
        runner=JobRunner(FakeAdapter(), tmp_path / "jobs", spawn=FakeSpawner(*procs)),
        downloads=DownloadManager(downloader),
        datasets=DatasetLibrary(tmp_path / "datasets"),
        recipes_dir=tmp_path / "recipes",
        jobs_root=tmp_path / "jobs",
        probe=lambda: RTX_3060,
    )
    # loopback base_url: the security middleware rejects non-loopback Hosts
    return TestClient(create_app(deps), base_url="http://127.0.0.1:8471"), deps


def recipe_json(tmp_path, **overrides) -> dict:
    return make_recipe(tmp_path, **overrides).model_dump(mode="json")


def ws_connect(client: TestClient, path: str):
    # the testclient stamps Host: testserver on WS upgrades; the security
    # middleware (rightly) refuses that, so pin a loopback Host explicitly
    return client.websocket_connect(path, headers={"host": "127.0.0.1:8471"})


_TERMINAL = ("completed", "completed_early", "failed", "cancelled")


def drain_ws(ws) -> list[dict]:
    """Receive until the terminal state event the stream guarantees."""
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["kind"] == "state" and event["state"] in _TERMINAL:
            return events


class HangingProcess:
    """Runs forever until terminated — pins the worker deterministically."""

    def __init__(self) -> None:
        self._released = asyncio.Event()
        self._rc: int | None = None
        self.terminated = False

    @property
    def returncode(self) -> int | None:
        return self._rc

    async def next_line(self) -> str | None:
        if not self.terminated:
            await self._released.wait()
        return None

    async def wait(self) -> int:
        self._rc = -15
        return self._rc

    def terminate(self) -> None:
        self.terminated = True
        self._released.set()

    kill = terminate


# ── Diagnose and models ──────────────────────────────────────────────────────


def test_diagnose_returns_hardware_and_capabilities(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        data = client.get("/diagnose").json()
    assert data["hardware"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 3060"
    verdicts = {m["model_key"]: m for m in data["capabilities"]["models"]}
    assert verdicts["sdxl"]["status"] == "available"
    assert verdicts["sdxl"]["preset_name"] is not None


def test_models_merges_capability_with_download_state(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        entries = client.get("/models").json()
    by_key = {e["capability"]["model_key"]: e for e in entries}
    assert by_key["sdxl"]["download_state"] == "not_downloaded"
    assert by_key["sdxl"]["capability"]["status"] == "available"

    with client:
        assert client.post("/models/nonsense/download").status_code == 404


def test_download_flow_streams_events_and_updates_state(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        started = client.post("/models/sdxl/download")
        assert started.status_code == 202
        assert started.json()["started"] is True
        with ws_connect(client, "/models/sdxl/events") as ws:
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["state"] in ("completed", "failed"):
                    break
        assert events[0]["state"] == "checking"
        assert events[-1]["state"] == "completed"
        assert events[-1]["result"]["model_path"].endswith("sd_xl_base_1.0.safetensors")
        by_key = {e["capability"]["model_key"]: e for e in client.get("/models").json()}
        assert by_key["sdxl"]["download_state"] == "downloaded"
        # idempotent restart
        assert client.post("/models/sdxl/download").json()["started"] is False


# ── Recipes ──────────────────────────────────────────────────────────────────


def test_validate_returns_human_errors_verbatim(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    bad_resolution = recipe_json(tmp_path)
    bad_resolution["dataset"]["resolution"] = 1000
    # cross-field checks only run on field-valid documents, so probe them separately
    bad_prompts = recipe_json(tmp_path)
    bad_prompts["train"]["sample_every_steps"] = 100
    bad_prompts["train"]["sample_prompts"] = []
    with client:
        result = client.post("/recipes/validate", json=bad_resolution).json()
        assert result["valid"] is False
        assert any("multiple of 64" in e for e in result["errors"])  # the actionable words
        result = client.post("/recipes/validate", json=bad_prompts).json()
        assert result["valid"] is False
        assert any("sample_prompts" in e for e in result["errors"])
        assert client.post("/recipes/validate", json=recipe_json(tmp_path)).json() == {
            "valid": True,
            "errors": [],
        }


def test_recipe_crud_roundtrip(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        assert client.get("/recipes").json() == []
        put = client.put("/recipes/tuxedo-cat", json=recipe_json(tmp_path))
        assert put.status_code == 200
        assert client.get("/recipes").json() == ["tuxedo-cat"]
        fetched = client.get("/recipes/tuxedo-cat").json()
        assert fetched["name"] == "test-run"
        assert client.delete("/recipes/tuxedo-cat").status_code == 204
        assert client.get("/recipes/tuxedo-cat").status_code == 404
        # names are constrained: no traversal, no separators
        assert client.get("/recipes/bad%20name").status_code == 400


# ── Datasets ─────────────────────────────────────────────────────────────────


def test_dataset_routes_flow(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    sources = [str(fake_photo(tmp_path / "src" / f"cat-{i}.png", seed=i)) for i in range(2)]
    with client:
        created = client.post("/datasets", json={"name": "cats"})
        assert created.status_code == 201
        assert client.get("/datasets").json() == ["cats"]
        assert client.post("/datasets", json={"name": "no/slashes"}).status_code == 400

        ingest = client.post("/datasets/cats/images", json={"sources": sources}).json()
        assert sorted(ingest["added"]) == ["cat-0.png", "cat-1.png"]

        summary = client.get("/datasets/cats").json()
        assert (summary["total"], summary["included"]) == (2, 2)
        assert {i["filename"] for i in summary["images"]} == {"cat-0.png", "cat-1.png"}
        assert all(i["has_caption"] is False for i in summary["images"])

        put = client.put(
            "/datasets/cats/captions/cat-0.png", json={"caption": "a cat, sitting"}
        )
        assert put.status_code == 200
        got = client.get("/datasets/cats/captions/cat-0.png").json()
        assert got["caption"] == "a cat, sitting"
        assert client.get("/datasets/cats/captions/ghost.png").status_code == 404

        trigger = client.post(
            "/datasets/cats/trigger-word", json={"trigger_word": "sks-cat"}
        ).json()
        assert trigger["updated"] == 2
        assert (
            client.get("/datasets/cats/captions/cat-0.png").json()["caption"]
            == "sks-cat, a cat, sitting"
        )

        assert client.get("/datasets/nope").status_code == 404
        assert client.delete("/datasets/cats").status_code == 204
        assert client.get("/datasets").json() == []


# ── Jobs ─────────────────────────────────────────────────────────────────────


def test_job_ws_delivers_replay_and_terminal(tmp_path) -> None:
    client, _ = make_client(tmp_path, procs=[FakeProcess(["step 1/2\n", "step 2/2\n"])])
    with client:
        job = client.post("/jobs", json=recipe_json(tmp_path)).json()
        assert job["state"] in ("queued", "preparing", "running", "completed")

        with ws_connect(client, f"/jobs/{job['id']}/events") as ws:
            events = drain_ws(ws)
        states = [e["state"] for e in events if e["kind"] == "state"]
        assert states == ["queued", "preparing", "running", "completed"]
        assert [e["progress"]["step"] for e in events if e["kind"] == "progress"] == [1, 2]

        # late subscriber: full replay again, still ends terminally
        with ws_connect(client, f"/jobs/{job['id']}/events") as ws:
            replay = drain_ws(ws)
        assert [e["state"] for e in replay if e["kind"] == "state"] == states

        record = client.get(f"/jobs/{job['id']}").json()
        assert record["state"] == "completed"
        assert client.get("/jobs").json()[0]["id"] == job["id"]


def test_cancel_from_queued_via_http(tmp_path) -> None:
    hanging = HangingProcess()
    client, _ = make_client(tmp_path, procs=[hanging])
    with client:
        first = client.post("/jobs", json=recipe_json(tmp_path)).json()
        second = client.post("/jobs", json=recipe_json(tmp_path)).json()

        cancelled = client.post(f"/jobs/{second['id']}/cancel").json()
        assert cancelled["state"] == "cancelled"
        states = [s["state"] for s in cancelled["state_history"]]
        assert states == ["queued", "cancelled"]  # never prepared, never spawned

        client.post(f"/jobs/{first['id']}/cancel")  # release the hanging worker
        with ws_connect(client, f"/jobs/{first['id']}/events") as ws:
            assert drain_ws(ws)[-1]["state"] == "cancelled"
    assert hanging.terminated

    with ws_connect(client, "/jobs/nope/events") as ws, pytest.raises(WebSocketDisconnect):
        ws.receive_json()  # server accepted, then closed 4004: unknown job


def test_multipart_upload_reuses_ingest_verdicts(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    photo = fake_photo(tmp_path / "src" / "cat.png", seed=5)
    other = fake_photo(tmp_path / "src" / "other.png", seed=6)
    with client:
        client.post("/datasets", json={"name": "cats"})
        files = [
            ("files", ("cat.png", photo.read_bytes(), "image/png")),
            ("files", ("cat-again.png", photo.read_bytes(), "image/png")),  # same bytes
            ("files", ("notes.jpg", b"not an image", "image/jpeg")),
            ("files", ("../sneaky.png", other.read_bytes(), "image/png")),  # path in name
        ]
        result = client.post("/datasets/cats/upload", files=files).json()
        assert result["added"] == ["cat.png", "sneaky.png"]  # traversal stripped to basename
        reasons = " | ".join(s["reason"] for s in result["skipped"])
        assert "exact duplicate" in reasons and "corrupted" in reasons

        summary = client.get("/datasets/cats").json()
        assert summary["included"] == 2
        assert client.post("/datasets/nope/upload", files=files[:1]).status_code == 404


def test_cancel_with_keep_finishes_completed_early(tmp_path) -> None:
    client, _ = make_client(
        tmp_path, procs=[HangingProcess()]
    )
    with client:
        job = client.post("/jobs", json=recipe_json(tmp_path)).json()
        with ws_connect(client, f"/jobs/{job['id']}/events") as ws:
            while True:  # wait until it is actually running
                if ws.receive_json().get("state") == "running":
                    break
            assert client.post(f"/jobs/{job['id']}/cancel?keep=true").status_code == 200
            while True:  # the worker finishes the transition asynchronously
                event = ws.receive_json()
                if event["kind"] == "state" and event["state"] in _TERMINAL:
                    break
        assert event["state"] == "completed_early"
        record = client.get(f"/jobs/{job['id']}").json()
        assert record["state"] == "completed_early"
        assert record["artifact"].endswith("lora.safetensors")


# ── Artifact ─────────────────────────────────────────────────────────────────


def test_job_artifact_404_until_present_then_served(tmp_path) -> None:
    from pathlib import Path

    client, _ = make_client(tmp_path, procs=[FakeProcess(["step 1/1\n"])])
    with client:
        job = client.post("/jobs", json=recipe_json(tmp_path)).json()
        with ws_connect(client, f"/jobs/{job['id']}/events") as ws:
            drain_ws(ws)  # wait for completion
        record = client.get(f"/jobs/{job['id']}").json()
        assert record["artifact"] is not None
        # record points at the artifact, but the fake engine wrote no file yet
        assert client.get(f"/jobs/{job['id']}/artifact").status_code == 404

        Path(record["artifact"]).write_bytes(b"LORA-WEIGHTS")
        served = client.get(f"/jobs/{job['id']}/artifact")
        assert served.status_code == 200
        assert served.content == b"LORA-WEIGHTS"

        assert client.get("/jobs/nope/artifact").status_code == 404


# ── Hostile-browser defenses ─────────────────────────────────────────────────


def test_dns_rebinding_host_is_refused(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        rebound = client.get("/models", headers={"Host": "evil.example"})
        assert rebound.status_code == 403
        assert "loopback" in rebound.text
        # loopback Hosts in any spelling keep working
        assert client.get("/models", headers={"Host": "localhost:8471"}).status_code == 200
        assert client.get("/models", headers={"Host": "[::1]:8471"}).status_code == 200


def test_cross_origin_writes_refused_plain_and_local_allowed(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    good = recipe_json(tmp_path)
    with client:
        # no Origin header (curl, Tauri shell): works
        assert client.post("/recipes/validate", json=good).status_code == 200
        # loopback and Tauri origins: work
        for origin in ("http://127.0.0.1:8471", "http://localhost:5173", "tauri://localhost"):
            response = client.post(
                "/recipes/validate", json=good, headers={"Origin": origin}
            )
            assert response.status_code == 200, origin
        # foreign origin on a state-changing request: refused
        evil = client.post(
            "/recipes/validate", json=good, headers={"Origin": "https://evil.example"}
        )
        assert evil.status_code == 403
        assert "cross-origin" in evil.text
        # foreign origin on a read: allowed (the browser's SOP guards the response)
        assert (
            client.get("/models", headers={"Origin": "https://evil.example"}).status_code == 200
        )


# ── Bind policy ──────────────────────────────────────────────────────────────


def test_refuses_non_loopback_bind_without_override() -> None:
    ensure_local_bind("127.0.0.1")
    ensure_local_bind("localhost")
    ensure_local_bind("::1")
    with pytest.raises(SystemExit, match="allow-remote"):
        ensure_local_bind("0.0.0.0")
    with pytest.raises(SystemExit, match="no authentication"):
        ensure_local_bind("192.168.1.20")
    ensure_local_bind("0.0.0.0", allow_remote=True)  # explicit override is honored
