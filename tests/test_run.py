"""The launcher: what a workspace holds, and what it is allowed to be called."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))

import run  # noqa: E402
from craftax_game import VARIANTS  # noqa: E402


def test_a_workspace_holds_the_prompt_and_two_shims(tmp_path):
    ws = run.make_workspace(tmp_path, "B4XPT", ["pixels"])
    assert sorted(p.name for p in ws.iterdir()) == ["CLAUDE.md", "act", "python"]
    brief = (ws / "CLAUDE.md").read_text()
    assert brief.startswith("You are playing a game you have never seen before.")
    assert "notes.md" in brief, "the playing doctrine did not travel"


def test_the_brief_says_which_channels_this_run_gives(tmp_path):
    """Written by hand it would drift from what the log carries the first time
    somebody ran with a different --obs."""
    pixels = (run.make_workspace(tmp_path / "a", "B4XPT", ["pixels"]) / "CLAUDE.md").read_text()
    assert "`frames/`" in pixels and "`text/`" not in pixels

    both = (run.make_workspace(tmp_path / "b", "K7M3Q", ["pixels", "text"]) / "CLAUDE.md").read_text()
    assert "`frames/`" in both and "`text/`" in both
    assert "more than one form" in both

    words = (run.make_workspace(tmp_path / "c", "ZZZZ9", ["text"]) / "CLAUDE.md").read_text()
    assert "`text/`" in words and "`frames/`" not in words


def test_the_brief_explains_no_mechanic(tmp_path):
    """The action list hands over half the tech tree and nothing can be done about
    that. The brief must not hand over the other half."""
    brief = (run.make_workspace(tmp_path, "B4XPT", ["pixels"]) / "CLAUDE.md").read_text().lower()
    # Whole words: the doctrine tells a session its files "survive" a compaction, and
    # a substring match would read that as the game explaining itself.
    for word in ("craftax", "crafter", "minecraft", "achievement", "craft", "pickaxe",
                 "wood", "stone", "hunger", "monster", "mine", "tech tree", "inventory"):
        assert not re.search(rf"\b{word}\b", brief), f"the brief says {word!r}"


def test_the_agents_interpreter_reads_observations_and_not_the_package(tmp_path):
    """The run is unplayable without numpy and Pillow, and compromised with craftax."""
    ws = run.make_workspace(tmp_path, "B4XPT", ["pixels"])
    ok = subprocess.run([str(ws / "python"), "-c", "import numpy, PIL; print('ok')"],
                        capture_output=True, text=True, timeout=120)
    assert ok.returncode == 0 and "ok" in ok.stdout, ok.stderr
    for module in ("craftax", "jax"):
        blocked = subprocess.run([str(ws / "python"), "-c", f"import {module}"],
                                 capture_output=True, text=True, timeout=120)
        assert blocked.returncode != 0, f"the agent's interpreter can import {module}"


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def test_labels_are_opaque_and_recorded_outside_every_workspace(tmp_path):
    plan = [("craftax", 0), ("craftax", 1), ("classic", 3)]
    known = run.assign_labels(tmp_path, plan)
    assert sorted(known.values()) == sorted(plan)
    for label in known:
        assert len(label) == run.LABEL_LEN
        assert set(label) <= set(run.LABEL_CHARS)
        assert not any(v in label.lower() for v in VARIANTS)
    # The mapping lives under .rig/, not the launch root: the root is one `..` from
    # every workspace.
    assert (tmp_path / run.RIG / run.LABELS).is_file()
    assert not (tmp_path / run.LABELS).exists()


def test_labels_are_stable_across_calls_and_extend(tmp_path):
    first = run.assign_labels(tmp_path, [("craftax", 0)])
    again = run.assign_labels(tmp_path, [("craftax", 0), ("classic", 0)])
    assert first.items() <= again.items(), "a second call renamed an existing world"
    assert len(again) == 2


def test_labels_are_not_derived_from_the_spec(tmp_path):
    """A name a session could invert would let it recognise a world it had played,
    and a launch running one world twice would have two workspaces announcing it."""
    seen = set()
    for i in range(6):
        root = tmp_path / str(i)
        seen |= set(run.assign_labels(root, [("craftax", 0)]))
    assert len(seen) > 1, "the label is a function of the world"


# --------------------------------------------------------------------------- #
# What to play
# --------------------------------------------------------------------------- #


def test_specs():
    def parsed(worlds, seed=0):
        return run.specs(type("A", (), {"worlds": worlds, "seed": seed})())

    assert parsed([]) == [("craftax", 0)]
    assert parsed([], seed=4) == [("craftax", 4)]
    assert parsed(["craftax", "craftax:1", "classic:3"]) == [
        ("craftax", 0), ("craftax", 1), ("classic", 3)
    ]
    assert parsed(["craftax", "craftax"]) == [("craftax", 0)], "a repeat is one world"
    with pytest.raises(SystemExit, match="not one of"):
        parsed(["nethack"])


def test_unfinished_reads_the_launch_record_not_the_workspace(tmp_path):
    run.assign_labels(tmp_path, [("craftax", 0), ("classic", 2)])
    known = json.loads((tmp_path / run.RIG / run.LABELS).read_text())
    done, going = list(known)
    (tmp_path / done).mkdir()
    (tmp_path / done / "state.json").write_text(json.dumps({"terminal": True}))
    (tmp_path / going).mkdir()
    (tmp_path / going / "state.json").write_text(json.dumps({"terminal": False}))

    left = run.unfinished(tmp_path)
    assert [label for label, _ in left] == [going]
    assert left[0][1] == (known[going]["variant"], known[going]["seed"])

    with pytest.raises(SystemExit, match="not a launch directory"):
        run.unfinished(tmp_path / "nowhere")


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


def test_the_verdict_is_kept_out_of_the_workspace(tmp_path):
    """It carries the achievement set by name, and a replay is told to read what its
    predecessor left behind."""
    report = run.Report(label="B4XPT", variant="craftax", seed=0,
                        workspace=str(tmp_path / "B4XPT"),
                        achievements=["collect_wood", "make_wood_pickaxe"], score=2)
    path = run.write_report(tmp_path, report)
    assert path == tmp_path / run.RIG / "reports" / "B4XPT.json"
    assert not (tmp_path / "B4XPT" / "report.json").exists()
    assert "collect_wood" in path.read_text()


def test_rotation_leaves_no_verdict_behind(tmp_path):
    """A replay must not overwrite the result of the session it is replaying."""
    first = run.Report(label="B4XPT", workspace="x", score=2)
    run.write_report(tmp_path, first)
    run.write_report(tmp_path, run.Report(label="B4XPT", workspace="x", score=9))
    out = tmp_path / run.RIG / "reports"
    assert json.loads((out / "B4XPT-attempt1.json").read_text())["score"] == 2
    assert json.loads((out / "B4XPT.json").read_text())["score"] == 9
