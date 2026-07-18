"""Dataset prep tests: pillow-generated images in tmp dirs. No network, no torch."""

import random
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from loraforge.datasets.captioner import StubCaptioner
from loraforge.datasets.library import DatasetLibrary
from loraforge.recipes.schema import Recipe


def fake_photo(path: Path, seed: int, size: tuple[int, int] = (640, 640), mark: bool = False):
    """Deterministic 'photo': seeded blobs give distinct perceptual hashes."""
    rng = random.Random(seed)
    img = Image.new("RGB", size, tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(img)
    for _ in range(12):
        x0, y0 = rng.randrange(size[0]), rng.randrange(size[1])
        box = (x0, y0, x0 + rng.randrange(60, 320), y0 + rng.randrange(60, 320))
        draw.ellipse(box, fill=tuple(rng.randrange(256) for _ in range(3)))
    if mark:  # small corner change: near-duplicate, not exact duplicate
        draw.rectangle((0, 0, 20, 20), fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


@pytest.fixture()
def library(tmp_path: Path) -> DatasetLibrary:
    lib = DatasetLibrary(tmp_path / "datasets")
    lib.create("cats")
    return lib


# ── Ingestion ────────────────────────────────────────────────────────────────


def test_ingest_copies_and_never_moves(library: DatasetLibrary, tmp_path: Path) -> None:
    sources = [fake_photo(tmp_path / "src" / f"cat-{i}.png", seed=i) for i in range(3)]
    result = library.ingest("cats", sources)
    assert sorted(result.added) == ["cat-0.png", "cat-1.png", "cat-2.png"]
    assert result.skipped == []
    for source in sources:
        assert source.exists()  # originals untouched
    assert len(library.status("cats").images) == 3


def test_ingest_skips_exact_duplicates_with_reason(
    library: DatasetLibrary, tmp_path: Path
) -> None:
    original = fake_photo(tmp_path / "src" / "cat.png", seed=1)
    same_bytes = tmp_path / "src" / "cat-copy.png"
    same_bytes.write_bytes(original.read_bytes())

    result = library.ingest("cats", [original, same_bytes])
    assert result.added == ["cat.png"]
    assert len(result.skipped) == 1
    assert "exact duplicate of 'cat.png'" in result.skipped[0].reason

    again = library.ingest("cats", [original])  # re-ingesting later is also a no-op
    assert again.added == [] and "duplicate" in again.skipped[0].reason


def test_ingest_rejects_junk_with_human_reasons(
    library: DatasetLibrary, tmp_path: Path
) -> None:
    not_an_image = tmp_path / "notes.jpg"
    not_an_image.write_text("actually a text file")
    unsupported = tmp_path / "clip.gif"
    unsupported.write_bytes(b"GIF89a")
    missing = tmp_path / "nope.png"

    result = library.ingest("cats", [not_an_image, unsupported, missing])
    assert result.added == []
    reasons = {Path(s.source).name: s.reason for s in result.skipped}
    assert "corrupted" in reasons["notes.jpg"]
    assert "not a supported image" in reasons["clip.gif"]
    assert reasons["nope.png"] == "file not found"


def test_ingest_transcodes_heic_for_engine_compatibility(
    library: DatasetLibrary, tmp_path: Path
) -> None:
    heic = fake_photo(tmp_path / "src" / "IMG_0042.heic", seed=7)
    result = library.ingest("cats", [heic])
    assert result.added == ["IMG_0042.jpg"]  # kohya can't read HEIC → stored as JPEG
    assert heic.exists()  # original untouched
    entry = library.status("cats").images[0]
    assert entry.included and (entry.width, entry.height) == (640, 640)

    again = library.ingest("cats", [heic])  # same photo again → same JPEG bytes → dup
    assert again.added == [] and "duplicate" in again.skipped[0].reason


def test_ingest_brings_existing_caption_sidecars(
    library: DatasetLibrary, tmp_path: Path
) -> None:
    source = fake_photo(tmp_path / "src" / "cat.png", seed=1)
    source.with_suffix(".txt").write_text("a cat, sitting\n")
    library.ingest("cats", [source])
    assert library.get_caption("cats", "cat.png") == "a cat, sitting"


# ── Quality checks and duplicates ────────────────────────────────────────────


def test_status_quality_checks_and_counts(library: DatasetLibrary, tmp_path: Path) -> None:
    directory = library.root / "cats"
    fake_photo(directory / "good.png", seed=1)
    fake_photo(directory / "tiny.png", seed=2, size=(200, 200))
    fake_photo(directory / "smallish.png", seed=3, size=(400, 400))
    (directory / "broken.png").write_bytes(b"not really a png")

    summary = library.status("cats")
    by_name = {i.filename: i for i in summary.images}

    assert by_name["good.png"].included and by_name["good.png"].warnings == []
    assert by_name["good.png"].width == 640

    assert not by_name["tiny.png"].included
    assert "too small" in by_name["tiny.png"].reason and "256" in by_name["tiny.png"].reason

    assert by_name["smallish.png"].included  # trains, but warned
    assert any("small image" in w for w in by_name["smallish.png"].warnings)

    assert not by_name["broken.png"].included
    assert "corrupted" in by_name["broken.png"].reason

    assert (summary.total, summary.included, summary.excluded) == (4, 2, 2)


def test_status_flags_near_duplicates_but_not_distinct_images(
    library: DatasetLibrary, tmp_path: Path
) -> None:
    directory = library.root / "cats"
    fake_photo(directory / "a-original.png", seed=1)
    fake_photo(directory / "b-retouched.jpg", seed=1, mark=True)  # same shot, tiny edit
    fake_photo(directory / "c-different.png", seed=2)

    summary = library.status("cats")
    by_name = {i.filename: i for i in summary.images}

    near = by_name["b-retouched.jpg"]
    assert near.included  # flagged, never auto-excluded
    assert any("nearly identical to 'a-original.png'" in w for w in near.warnings)
    assert by_name["a-original.png"].warnings == []
    assert by_name["c-different.png"].warnings == []


# ── Captions and trigger word ────────────────────────────────────────────────


def test_caption_roundtrip_and_trigger_injection(
    library: DatasetLibrary, tmp_path: Path
) -> None:
    directory = library.root / "cats"
    fake_photo(directory / "one.png", seed=1)
    fake_photo(directory / "two.png", seed=2)
    fake_photo(directory / "three.png", seed=3)

    library.set_caption("cats", "one.png", "a cat, sitting on a sofa")
    library.set_caption("cats", "two.png", "photo of sks-cat, outdoors")  # already tagged
    assert library.get_caption("cats", "one.png") == "a cat, sitting on a sofa"
    assert library.get_caption("cats", "three.png") is None

    updated = library.inject_trigger_word("cats", "sks-cat")
    assert updated == 2  # one.png prepended, three.png created; two.png untouched
    assert library.get_caption("cats", "one.png") == "sks-cat, a cat, sitting on a sofa"
    assert library.get_caption("cats", "two.png") == "photo of sks-cat, outdoors"
    assert library.get_caption("cats", "three.png") == "sks-cat"

    assert library.inject_trigger_word("cats", "sks-cat") == 0  # idempotent

    with pytest.raises(FileNotFoundError, match="no image named"):
        library.get_caption("cats", "ghost.png")
    with pytest.raises(ValueError, match="invalid image filename"):
        library.get_caption("cats", "../../etc/passwd")


def test_summary_path_is_what_a_recipe_references(
    library: DatasetLibrary, tmp_path: Path
) -> None:
    fake_photo(library.root / "cats" / "one.png", seed=1)
    summary = library.status("cats")
    recipe = Recipe.model_validate(
        {
            "name": "cats-lora",
            "model": "sdxl",
            "dataset": {"path": str(summary.path)},
            "train": {"sample_every_steps": 0},
        }
    )
    assert recipe.dataset.path == summary.path


# ── Captioner stub ───────────────────────────────────────────────────────────


def test_stub_captioner_speaks_human_until_real_one_lands(tmp_path: Path) -> None:
    stub = StubCaptioner()
    problems = stub.check_environment(tmp_path)
    assert len(problems) == 1 and "auto-captioning is not available" in problems[0]
    with pytest.raises(NotImplementedError, match="captions by hand"):
        stub.compile(tmp_path, tmp_path)
    assert stub.parse_line("anything") is None
    assert stub.collect(tmp_path) == {}
