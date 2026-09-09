"""The launcher: what a workspace holds, and what it is allowed to be called."""

import asyncio
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


def test_the_brief_says_what_the_run_is_judged_on(tmp_path):
    """The first pilot was told what it could do and not what any of it was for, so it
    maximised the only number that grew: reward summed over the run, which counts one
    achievement once per life. Naming the unit the benchmark itself reports — return
    on a single episode — gives away nothing about the world."""
    brief = (run.make_workspace(tmp_path, "B4XPT", ["pixels"]) / "CLAUDE.md").read_text()
    # Wrapped prose: the sentence is there but the line breaks are not where a naive
    # substring search would put them.
    said = " ".join(brief.split())
    assert "judged on is the best single life in it" in said
    assert "a total across lives is not the thing to grow" in said


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


# --------------------------------------------------------------------------- #
# Carrying a run on
# --------------------------------------------------------------------------- #


def test_continuable_is_the_opposite_question_to_unfinished(tmp_path):
    """`--replay` asks which worlds never reached an ending, and starts those over.
    `--continue` asks which have played anything at all, because a run that spent
    its budget is exactly the one worth extending."""
    run.assign_labels(tmp_path, [("craftax", 0), ("craftax", 1), ("classic", 2)])
    known = json.loads((tmp_path / run.RIG / run.LABELS).read_text())
    spent, going, untouched = list(known)
    for label, state in (
        (spent, {"terminal": True, "outcome": "budget", "actions_used": 3000}),
        (going, {"terminal": False, "actions_used": 412}),
        (untouched, {"terminal": False, "actions_used": 0}),
    ):
        (tmp_path / label).mkdir()
        (tmp_path / label / "state.json").write_text(json.dumps(state))

    assert [label for label, _ in run.unfinished(tmp_path)] == [going, untouched]
    assert sorted(label for label, _ in run.continuable(tmp_path)) == sorted([spent, going])
    assert run.continuable(tmp_path)[0][1] == (
        known[run.continuable(tmp_path)[0][0]]["variant"],
        known[run.continuable(tmp_path)[0][0]]["seed"],
    )
    with pytest.raises(SystemExit, match="not a launch directory"):
        run.continuable(tmp_path / "nowhere")


def test_played_reads_the_open_record(tmp_path):
    assert run.played(tmp_path / "nothing here") == 0
    (tmp_path / "W").mkdir()
    (tmp_path / "W" / "state.json").write_text(json.dumps({"actions_used": 7}))
    assert run.played(tmp_path / "W") == 7


def test_the_continuation_task_hands_over_a_run_and_not_a_world():
    """REPLAY_TASK's world went back to the beginning and only the notes carried
    over. This one carries the run itself, so what it must not say is that anything
    has been reset — and what it must say is that the log is too big to read and
    that this session has a successor of its own."""
    said = " ".join(run.CONTINUE_TASK.split())
    assert "already part-played" in said
    assert "exactly where the last session left it" in said
    assert "too big to read" in said
    assert "leave notes.md fit for whoever comes next" in said
    assert "beginning" not in said and "started again" not in said
    assert "reports this session is over" in said, "it would play past its stint"


def test_a_kept_workspace_gets_the_current_brief(tmp_path):
    """`--replay` and `--continue` both keep the workspace a previous launch built,
    and with it whatever CLAUDE.md the harness wrote then. A brief that contradicts
    the log is worse than a stale one."""
    ws = run.make_workspace(tmp_path, "B4XPT", ["pixels"])
    (ws / "state.json").write_text(json.dumps({"obs": ["text"]}))
    (ws / "notes.md").write_text("what I worked out")
    (ws / "CLAUDE.md").write_text("a brief from an older harness")

    run.refresh_brief(ws)
    now = (ws / "CLAUDE.md").read_text()
    # From the record, not from how this workspace happened to be built.
    assert "`text/`" in now and "`frames/`" not in now
    assert "stint" in now, "the prompt that travelled did not carry the handover"
    assert (ws / "notes.md").read_text() == "what I worked out"


def test_the_prompt_says_a_stint_is_not_the_whole_budget():
    """The pilot ground `reset` because the interface offered a lever the game did
    not. A prompt that told a stinted session the budget was all its own would be
    the same mistake in the other direction — it would spend a successor's actions
    on the assumption they were about to be lost."""
    said = " ".join((run.REPO / "PROMPT.md").read_text().split())
    assert "The budget belongs to the run" in said
    assert "another continues the same run from exactly where you stopped" in said
    assert "a finding you did not write down did not happen" in said
    assert "the budget is the whole of what you have" not in said


def test_a_run_without_a_stint_is_one_session(tmp_path, monkeypatch):
    """Chaining is what `--stint` turns on. Without one, a session that stops with
    budget left has decided it is finished, and the launcher does not overrule it —
    which is how every run behaved before stints existed."""
    starts, ws = [], tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["pixels"], "actions_used": 0, "terminal": False}))

    async def fake_act(where, *argv):
        if argv[0] == "init":
            starts.append(argv)
            played = json.loads((ws / "state.json").read_text())
            played["actions_used"] += 10
            (ws / "state.json").write_text(json.dumps(played))
        return ""

    async def fake_session(argv, where, env, report, agent):
        return "", 0

    monkeypatch.setattr(run, "act", fake_act)
    monkeypatch.setattr(run, "one_session", fake_session)
    monkeypatch.setattr(run, "refresh_brief", lambda w: None)
    monkeypatch.setattr(run, "session_config", lambda r, label: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "max_score", "best_episode", "best_episode_pct",
                        "episodes_completed", "mean_episode", "mean_episode_pct",
                        "score", "score_pct", "lives", "deaths", "unique_cells",
                        "max_level")} | {"episode_scores": [], "achievements": []}))

    def go(stint, max_sessions=3):
        starts.clear()
        args = type("A", (), {
            "carry_on": str(tmp_path), "replay": None, "obs": ["pixels"], "budget": 500,
            "stint": stint, "max_sessions": max_sessions, "fresh_world": False,
            "agent": "claude", "model": "m", "dry_run": False,
        })()
        report = asyncio.run(run.play("W", ("craftax", 0), tmp_path, args, {}, ["W"],
                                      asyncio.Semaphore(1)))
        return report, list(starts)

    report, opened = go(stint=0)
    assert report.sessions == 1 and len(opened) == 1

    report, opened = go(stint=10)
    assert report.sessions == 3, "a stinted run stopped chaining"
    assert all("--resume" in one for one in opened), "a chained session restarted the world"
    assert all("--stint" in one for one in opened)
