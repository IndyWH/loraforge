"""Job runner tests: state machine, OOM step-down loop, cancellation, records.

No GPU, no real subprocesses: a fake adapter parses scripted stdout lines
from a fake process. Async scenarios run under plain asyncio.run().
"""

import asyncio
import json
import re
from pathlib import Path

import pytest

from loraforge.engines.base import LaunchPlan, ProgressEvent, TrainResult
from loraforge.jobs.runner import JobEvent, JobRunner, JobState, SubmitRefused
from loraforge.jobs.stepdown import step_down
from loraforge.recipes.schema import Recipe

# ── Fixtures-by-hand ─────────────────────────────────────────────────────────


def make_recipe(tmp_path: Path, **overrides) -> Recipe:
    recipe = Recipe.model_validate(
        {
            "name": "test-run",
            "model": "sdxl",
            "dataset": {"path": str(tmp_path / "images"), "resolution": 1024},
            "train": {"sample_every_steps": 0},
        }
    )
    return recipe.with_overrides(overrides) if overrides else recipe


class FakeAdapter:
    """EngineAdapter-shaped test double with a trivial line protocol."""

    name = "fake"

    def __init__(self) -> None:
        self.compiled: list[Recipe] = []

    def check_environment(self, env_dir: Path) -> list[str]:
        return []

    def compile(self, recipe: Recipe, workdir: Path) -> LaunchPlan:
        self.compiled.append(recipe)
        return LaunchPlan(
            argv=["train", f"--res={recipe.dataset.resolution}"],
            cwd=workdir,
            config_files={workdir / "dataset.toml": f"resolution = {recipe.dataset.resolution}"},
        )

    def parse_line(self, line: str) -> ProgressEvent | None:
        if "out of memory" in line:
            return ProgressEvent(is_oom=True, message=line.strip())
        if m := re.match(r"step (\d+)/(\d+)", line):
            return ProgressEvent(step=int(m[1]), total_steps=int(m[2]))
        return None

    def collect(self, workdir: Path) -> TrainResult:
        return TrainResult(
            artifact=workdir / "lora.safetensors", format="kohya", logs=workdir / "logs"
        )


class FakeProcess:
    """Scripted stdout; yields to the event loop before each line."""

    def __init__(self, lines: list[str], returncode: int = 0) -> None:
        self._lines = list(lines)
        self._final_rc = returncode
        self._rc: int | None = None
        self.terminated = False
        self.killed = False

    @property
    def returncode(self) -> int | None:
        return self._rc

    async def next_line(self) -> str | None:
        await asyncio.sleep(0)  # let subscribers (and cancel()) interleave
        if self.terminated or self.killed or not self._lines:
            return None
        return self._lines.pop(0)

    async def wait(self) -> int:
        if self._rc is None:
            self._rc = -15 if (self.terminated or self.killed) else self._final_rc
        return self._rc

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class FakeSpawner:
    def __init__(self, *procs: FakeProcess) -> None:
        self.procs = list(procs)
        self.launched: list[LaunchPlan] = []

    async def __call__(self, plan: LaunchPlan) -> FakeProcess:
        self.launched.append(plan)
        return self.procs.pop(0)


async def drain(runner: JobRunner, job_id: str) -> list[JobEvent]:
    """Consume a job's event stream to its guaranteed terminal event."""
    return [event async for event in runner.events(job_id)]


def states(events: list[JobEvent]) -> list[str]:
    return [e.state.value for e in events if e.kind == "state"]


def record_of(runner: JobRunner, job_id: str) -> dict:
    return json.loads(runner.get(job_id).record_path.read_text(encoding="utf-8"))


# ── Happy path ───────────────────────────────────────────────────────────────


def test_happy_path_states_progress_and_record(tmp_path: Path) -> None:
    async def scenario() -> None:
        adapter = FakeAdapter()
        spawner = FakeSpawner(FakeProcess(["prep\n", "step 1/10\n", "step 10/10\n"]))
        runner = JobRunner(adapter, tmp_path, spawn=spawner)
        job = await runner.submit(make_recipe(tmp_path))
        events = await asyncio.wait_for(drain(runner, job.id), timeout=5)
        await runner.close()

        assert states(events) == ["queued", "preparing", "running", "completed"]
        assert events[-1].is_terminal
        assert [e.progress.step for e in events if e.kind == "progress"] == [1, 10]

        # config_files written during preparing, raw lines logged
        assert (job.workdir / "dataset.toml").read_text() == "resolution = 1024"
        assert "prep" in job.log_path.read_text()

        # the on-disk record mirrors the event history the subscriber saw
        record = record_of(runner, job.id)
        assert record["state"] == "completed"
        assert record["artifact"].endswith("lora.safetensors")
        assert [s["state"] for s in record["state_history"]] == states(events)

    asyncio.run(scenario())


# ── OOM step-down loop ───────────────────────────────────────────────────────


def test_oom_steps_down_and_succeeds(tmp_path: Path) -> None:
    async def scenario() -> None:
        adapter = FakeAdapter()
        first = FakeProcess(["step 1/10\n", "CUDA out of memory\n", "step 2/10\n"])
        spawner = FakeSpawner(first, FakeProcess(["step 10/10\n"]))
        runner = JobRunner(adapter, tmp_path, spawn=spawner)
        job = await runner.submit(make_recipe(tmp_path, model="flux_dev"))
        events = await asyncio.wait_for(drain(runner, job.id), timeout=5)
        await runner.close()

        assert states(events) == [
            "queued", "preparing", "running", "oom_stepdown", "preparing", "running", "completed",
        ]
        assert first.terminated  # the OOM'd process tree was shut down
        assert adapter.compiled[-1].train.blocks_to_swap == 18  # flux: block swap first
        # attempt 1 stops at the OOM line ("step 2" is never parsed); attempt 2 runs to the end
        steps = [e.progress.step for e in events if e.kind == "progress" and e.progress.step]
        assert steps == [1, 10]

        stepdown_event = next(e for e in events if e.state is JobState.OOM_STEPDOWN)
        assert "ran out of memory" in stepdown_event.message
        assert "RAM" in stepdown_event.message  # says what changed, in human words

        record = record_of(runner, job.id)
        assert record["state"] == "completed"
        assert len(record["stepdowns"]) == 1
        assert record["stepdowns"][0]["from"]["blocks_to_swap"] == 0
        assert record["stepdowns"][0]["to"]["blocks_to_swap"] == 18
        assert record["recipe"]["train"]["blocks_to_swap"] == 18  # snapshot is the final recipe

    asyncio.run(scenario())


def test_oom_retries_exhausted_fails_in_human_words(tmp_path: Path) -> None:
    async def scenario() -> None:
        adapter = FakeAdapter()
        oom = ["CUDA out of memory\n"]
        spawner = FakeSpawner(FakeProcess(oom), FakeProcess(oom), FakeProcess(oom))
        runner = JobRunner(adapter, tmp_path, spawn=spawner, max_retries=2)
        job = await runner.submit(make_recipe(tmp_path))  # sdxl: resolution notches
        events = await asyncio.wait_for(drain(runner, job.id), timeout=5)
        await runner.close()

        assert job.state is JobState.FAILED
        assert states(events).count("oom_stepdown") == 2
        assert [r["dataset"]["resolution"] for r in [job.recipe.model_dump(mode="json")]] == [512]
        # the failure explains everything that was tried, in plain language
        assert "out of memory" in job.error
        assert "1024 to 768" in job.error and "768 to 512" in job.error
        assert "Close other applications" in job.error
        assert record_of(runner, job.id)["error"] == job.error

    asyncio.run(scenario())


def test_oom_with_nothing_left_to_try_fails_immediately(tmp_path: Path) -> None:
    async def scenario() -> None:
        adapter = FakeAdapter()
        spawner = FakeSpawner(FakeProcess(["CUDA out of memory\n"]))
        runner = JobRunner(adapter, tmp_path, spawn=spawner)
        recipe = make_recipe(tmp_path, **{"dataset.resolution": 512, "train.batch_size": 1})
        job = await runner.submit(recipe)
        await asyncio.wait_for(drain(runner, job.id), timeout=5)
        await runner.close()

        assert job.state is JobState.FAILED
        assert "no tighter settings" in job.error
        assert len(spawner.launched) == 1  # no pointless retry

    asyncio.run(scenario())


# ── Cancellation ─────────────────────────────────────────────────────────────


def test_cancel_while_running_stops_the_process_tree(tmp_path: Path) -> None:
    async def scenario() -> None:
        adapter = FakeAdapter()
        proc = FakeProcess([f"step {i}/100\n" for i in range(1, 100)])
        runner = JobRunner(adapter, tmp_path, spawn=FakeSpawner(proc))
        job = await runner.submit(make_recipe(tmp_path))

        events = []
        async for event in runner.events(job.id):
            events.append(event)
            if event.kind == "progress" and event.progress.step == 2:
                await runner.cancel(job.id)
        await runner.close()

        assert job.state is JobState.CANCELLED
        assert events[-1].is_terminal
        assert proc.terminated
        assert record_of(runner, job.id)["state"] == "cancelled"

    asyncio.run(scenario())


def test_cancel_while_queued_never_launches(tmp_path: Path) -> None:
    async def scenario() -> None:
        adapter = FakeAdapter()
        spawner = FakeSpawner(FakeProcess(["step 10/10\n"]))
        runner = JobRunner(adapter, tmp_path, spawn=spawner)
        first = await runner.submit(make_recipe(tmp_path))
        second = await runner.submit(make_recipe(tmp_path))
        await runner.cancel(second.id)

        first_events = await asyncio.wait_for(drain(runner, first.id), timeout=5)
        second_events = await asyncio.wait_for(drain(runner, second.id), timeout=5)
        await runner.close()

        assert states(first_events)[-1] == "completed"
        assert states(second_events) == ["queued", "cancelled"]  # straight to terminal
        assert len(spawner.launched) == 1  # second job never spawned a process
        assert record_of(runner, second.id)["state"] == "cancelled"

    asyncio.run(scenario())


def test_stop_and_keep_collects_newest_checkpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        adapter = FakeAdapter()
        proc = FakeProcess([f"step {i}/100\n" for i in range(1, 100)])
        runner = JobRunner(adapter, tmp_path, spawn=FakeSpawner(proc))
        job = await runner.submit(make_recipe(tmp_path))

        events = []
        async for event in runner.events(job.id):
            events.append(event)
            if event.kind == "progress" and event.progress.step == 3:
                await runner.cancel(job.id, keep=True)
        await runner.close()

        assert job.state is JobState.COMPLETED_EARLY
        assert proc.terminated  # the process was still stopped
        assert job.artifact is not None and job.artifact.name == "lora.safetensors"
        terminal = events[-1]
        assert terminal.is_terminal and "kept the latest saved LoRA" in terminal.message
        record = record_of(runner, job.id)
        assert record["state"] == "completed_early"
        assert record["artifact"].endswith("lora.safetensors")

    asyncio.run(scenario())


def test_stop_and_keep_without_checkpoint_falls_back_to_cancelled(tmp_path: Path) -> None:
    class NoArtifactAdapter(FakeAdapter):
        def collect(self, workdir: Path):
            raise FileNotFoundError("no .safetensors artifact")

    async def scenario() -> None:
        runner = JobRunner(
            NoArtifactAdapter(),
            tmp_path,
            spawn=FakeSpawner(FakeProcess([f"step {i}/100\n" for i in range(1, 100)])),
        )
        job = await runner.submit(make_recipe(tmp_path))
        async for event in runner.events(job.id):
            if event.kind == "progress" and event.progress.step == 2:
                await runner.cancel(job.id, keep=True)
        await runner.close()

        assert job.state is JobState.CANCELLED
        assert "nothing to keep" in record_of(runner, job.id)["state_history"][-1]["message"]

    asyncio.run(scenario())


# ── Other failure paths ──────────────────────────────────────────────────────


def test_nonzero_exit_fails_and_points_at_the_log(tmp_path: Path) -> None:
    async def scenario() -> None:
        adapter = FakeAdapter()
        spawner = FakeSpawner(FakeProcess(["Traceback: ValueError\n"], returncode=1))
        runner = JobRunner(adapter, tmp_path, spawn=spawner)
        job = await runner.submit(make_recipe(tmp_path))
        await asyncio.wait_for(drain(runner, job.id), timeout=5)
        await runner.close()

        assert job.state is JobState.FAILED
        assert "code 1" in job.error and "job.log" in job.error
        assert "Traceback" in job.log_path.read_text()

    asyncio.run(scenario())


def test_late_subscriber_replays_history_and_terminates(tmp_path: Path) -> None:
    async def scenario() -> None:
        adapter = FakeAdapter()
        runner = JobRunner(adapter, tmp_path, spawn=FakeSpawner(FakeProcess(["step 5/5\n"])))
        job = await runner.submit(make_recipe(tmp_path))
        await asyncio.wait_for(drain(runner, job.id), timeout=5)  # job finishes
        replay = await asyncio.wait_for(drain(runner, job.id), timeout=5)  # subscribe after
        await runner.close()

        assert states(replay)[-1] == "completed"
        assert [e.seq for e in replay] == sorted({e.seq for e in replay})  # no dupes, in order

    asyncio.run(scenario())


# ── Step-down ladder (sync unit tests) ───────────────────────────────────────


def test_ladder_full_order_flux(tmp_path: Path) -> None:
    # block swap (18 → 34) → gradient checkpointing → resolution notch
    recipe = make_recipe(
        tmp_path, **{"model": "flux_dev", "train.gradient_checkpointing": False}
    )
    one = step_down(recipe)
    assert one.recipe.train.blocks_to_swap == 18
    assert one.recipe.train.gradient_checkpointing is False  # one rung at a time
    two = step_down(one.recipe)
    assert two.recipe.train.blocks_to_swap == 34
    three = step_down(two.recipe)  # blocks exhausted → checkpointing before resolution
    assert three.recipe.train.gradient_checkpointing is True
    assert three.recipe.dataset.resolution == 1024
    assert "checkpointing" in three.message
    four = step_down(three.recipe)
    assert four.recipe.dataset.resolution == 768
    assert "768" in four.message


def test_ladder_full_order_sdxl(tmp_path: Path) -> None:
    # no block swap for sdxl: checkpointing → resolution x2 → halve batch x2 → exhausted
    recipe = make_recipe(
        tmp_path, **{"train.gradient_checkpointing": False, "train.batch_size": 4}
    )
    one = step_down(recipe)
    assert one.recipe.train.gradient_checkpointing is True
    assert one.recipe.dataset.resolution == 1024  # quality untouched by the speed-only rung
    assert one.recipe.train.blocks_to_swap == 0
    two = step_down(one.recipe)
    assert two.recipe.dataset.resolution == 768
    three = step_down(two.recipe)
    assert three.recipe.dataset.resolution == 512
    four = step_down(three.recipe)  # notches exhausted → halve batch
    assert four.recipe.train.batch_size == 2
    five = step_down(four.recipe)
    assert five.recipe.train.batch_size == 1
    assert step_down(five.recipe) is None  # ladder exhausted


def test_ladder_block_swap_respects_batch_size_rule(tmp_path: Path) -> None:
    # schema forbids blocks_to_swap with batch_size > 2; the rung must handle it
    recipe = make_recipe(tmp_path, **{"model": "flux_dev", "train.batch_size": 4})
    stepped = step_down(recipe)
    assert stepped.recipe.train.blocks_to_swap == 18
    assert stepped.recipe.train.batch_size == 2
    assert "atch size" in stepped.message  # tells the user about the extra change


# ── Rule 5 at the runner: refuse early, never leak raw errors ────────────────


class BrokenEnvAdapter(FakeAdapter):
    """An adapter whose engine environment is not installed."""

    def check_environment(self, env_dir: Path | None = None) -> list[str]:
        return ["engine environment incomplete: accelerate missing"]


class MissingModelAdapter(FakeAdapter):
    def compile(self, recipe: Recipe, workdir: Path) -> LaunchPlan:
        raise ValueError(
            "base model for 'sdxl' has not been downloaded yet — "
            "run the model download step first"
        )


def test_submit_refused_before_any_job_when_engine_missing(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = JobRunner(BrokenEnvAdapter(), tmp_path / "jobs", spawn=FakeSpawner())
        with pytest.raises(SubmitRefused) as excinfo:
            await runner.submit(make_recipe(tmp_path))
        message = str(excinfo.value)
        assert "isn't set up" in message
        assert "loraforge setup" in message  # names the fix
        assert not list((tmp_path / "jobs").glob("*"))  # refused BEFORE job creation

    asyncio.run(scenario())


def test_submit_refused_when_model_files_missing(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = JobRunner(MissingModelAdapter(), tmp_path / "jobs", spawn=FakeSpawner())
        with pytest.raises(SubmitRefused) as excinfo:
            await runner.submit(make_recipe(tmp_path))
        assert "download" in str(excinfo.value)  # compile's human message, verbatim
        assert not list((tmp_path / "jobs").glob("*"))

    asyncio.run(scenario())


def test_spawn_failure_speaks_human_and_logs_the_raw_error(tmp_path: Path) -> None:
    async def exploding_spawn(plan: LaunchPlan) -> FakeProcess:
        raise FileNotFoundError(2, "No such file or directory", plan.argv[0])

    async def scenario() -> None:
        runner = JobRunner(FakeAdapter(), tmp_path, spawn=exploding_spawn)
        job = await runner.submit(make_recipe(tmp_path))
        events = await asyncio.wait_for(drain(runner, job.id), timeout=5)
        await runner.close()

        final = events[-1]
        assert final.state is JobState.FAILED
        assert "loraforge setup" in final.message  # the next step, in the message
        assert "Errno" not in final.message  # raw errno never reaches the UI
        assert "Errno" in job.log_path.read_text(encoding="utf-8")  # ...but is kept

    asyncio.run(scenario())


def test_unexpected_error_is_wrapped_and_logged(tmp_path: Path) -> None:
    class ExplodingAdapter(FakeAdapter):
        def parse_line(self, line: str) -> ProgressEvent | None:
            raise RuntimeError("[Errno 2] No such file or directory: 'weights.bin'")

    async def scenario() -> None:
        runner = JobRunner(
            ExplodingAdapter(), tmp_path, spawn=FakeSpawner(FakeProcess(["step 1/10\n"]))
        )
        job = await runner.submit(make_recipe(tmp_path))
        events = await asyncio.wait_for(drain(runner, job.id), timeout=5)
        await runner.close()

        final = events[-1]
        assert final.state is JobState.FAILED
        assert "Errno" not in final.message
        assert "not your settings" in final.message  # what happened, human-worded
        assert "loraforge diagnose" in final.message  # and what to do next
        assert str(job.log_path) in final.message  # where the raw detail lives
        assert "Errno 2" in job.log_path.read_text(encoding="utf-8")

    asyncio.run(scenario())


def test_engine_diagnosis_replaces_bare_exit_code(tmp_path: Path) -> None:
    class HintingAdapter(FakeAdapter):
        def parse_line(self, line: str) -> ProgressEvent | None:
            if "multiarray failed" in line:
                return ProgressEvent(
                    message=line.strip(),
                    fatal_hint=(
                        "The training engine's Python packages are mismatched — "
                        "run `loraforge setup` to repair them."
                    ),
                )
            return super().parse_line(line)

    async def scenario() -> None:
        doomed = FakeProcess(
            ["ImportError: numpy.core.multiarray failed to import\n"], returncode=1
        )
        runner = JobRunner(HintingAdapter(), tmp_path, spawn=FakeSpawner(doomed))
        job = await runner.submit(make_recipe(tmp_path))
        events = await asyncio.wait_for(drain(runner, job.id), timeout=5)
        await runner.close()

        final = events[-1]
        assert final.state is JobState.FAILED
        assert "loraforge setup" in final.message  # diagnosis, not a bare exit code
        assert "exited with code" not in final.message
        assert str(job.log_path) in final.message  # raw lines still findable

    asyncio.run(scenario())
