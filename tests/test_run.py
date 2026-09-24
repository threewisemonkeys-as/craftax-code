"""The launcher: what a workspace holds, and what it is allowed to be called."""

import asyncio
import json
import re
import sqlite3
import subprocess
import time
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))

from agents import AGENTS, HARVESTED  # noqa: E402

import run  # noqa: E402
from craftax_game import VARIANTS  # noqa: E402


def said(path: Path) -> str:
    """A brief's text with its line breaks taken out. `brief()` rewraps the
    paragraphs it fills in, so where a phrase breaks depends on the words around it,
    and a check for the phrase should not."""
    return " ".join(path.read_text().split())


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


def test_the_brief_says_what_a_death_costs_and_it_matches_the_run(tmp_path):
    """The one sentence a session can act on between lives.

    A run that replays one world rewards remembering it — a route, a map, a saved
    opening — and a fresh world punishes exactly that. Both briefs existed as one
    hardcoded sentence until a fresh-world run was asked for, which would have
    been told the world repeats while it did not. Tie each wording to the setting so
    the two cannot come apart again.

    Both settings are passed explicitly. The default moved to fresh on 2026-09-24,
    and a test that took either wording from the default would have gone on passing
    while saying nothing about the case it was named for.
    """
    kept = said(run.make_workspace(tmp_path / "k", "B4XPT", ["pixels"], False) / "CLAUDE.md")
    dealt = said(run.make_workspace(tmp_path / "f", "K7M3Q", ["pixels"], True) / "CLAUDE.md")
    assert "starts again from its beginning" in kept
    assert "a new world is dealt" in dealt
    assert "starts again from its beginning" not in dealt
    assert "{death}" not in kept and "{death}" not in dealt, "placeholder left unfilled"

    # The whole assembled CLAUDE.md, not just the generated half. PROMPT.md carried
    # "the same actions from the beginning do the same things" as fixed doctrine
    # until 2026-09-24 — under a fresh world that is false, and it is also a recipe
    # for the replay strategy that stops a run being played. Claims about what
    # survives a death belong to the brief, which knows the setting.
    for claim in ("the same one", "still holds", "the same actions from the beginning"):
        assert claim not in dealt, f"a fresh-world brief still promises {claim!r}"


def test_a_run_deals_a_new_world_each_life_unless_told_otherwise(tmp_path):
    """The default moved from one replayed world to a fresh one on 2026-09-24.

    A deterministic world is not the same benchmark played repeatedly — it is a
    benchmark that stops being played. MP374 wrote itself a `replay.py` at action
    5,753 and spent 69% of a 30,000-action budget re-executing a recorded life;
    its last life made five decisions in 3,681 actions, and its union finished
    three points above its best single life. Fresh worlds price that out.

    The rule has to hold in three places at once, or a run is told one thing and
    given another: the record's default, the actuator's CLI, and the brief.
    """
    import act  # noqa: PLC0415

    assert act.RunState().fresh_world is True, "the record's default"
    parser = act.build_parser()
    assert parser.parse_args(["init"]).fresh_world is True, "the actuator's default"
    assert parser.parse_args(["init", "--same-world"]).fresh_world is False
    assert parser.parse_args(["serve", "--same-world"]).fresh_world is False, "and the daemon's"
    with pytest.raises(SystemExit):
        parser.parse_args(["init", "--fresh-world", "--same-world"])

    brief = said(run.make_workspace(tmp_path, "B4XPT", ["pixels"]) / "CLAUDE.md")
    assert "a new world is dealt" in brief, "what a default run tells its session"


def test_a_refreshed_brief_takes_the_world_rule_from_the_record(tmp_path):
    """`--continue` passes no `--fresh-world`, so a refreshed brief that read the
    arguments would flip the rule under a run halfway through it. It reads the
    record, as it already does for the channels."""
    ws = run.make_workspace(tmp_path, "B4XPT", ["pixels"], True)
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["pixels"], "fresh_world": True, "actions_used": 0, "terminal": False}))
    run.refresh_brief(ws)
    assert "a new world is dealt" in said(ws / "CLAUDE.md")


def test_the_brief_is_wrapped_like_the_file_it_comes_from():
    """Every placeholder sits inside a hand-wrapped paragraph, so a filled one left
    a line of 110 characters in a file wrapped at 86, and the channel list 250. Every
    combination the brief is built from, because each one fills in something else."""
    for fresh_world in (True, False):
        for channels in (["pixels"], list(run.CHANNELS)):
            text = run.brief(channels, fresh_world)
            long = [line for line in text.splitlines() if len(line) > run.WIDTH]
            assert not long, (fresh_world, channels, long)


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


def test_a_continued_run_carries_what_it_already_spent(tmp_path):
    """A run outlives a launch as well as a session, and the bill is the run's.

    Everything the report takes from the record is already cumulative, so a launch
    that counted only its own turns and dollars described a 20,000-action run with
    one launch's costs.
    """
    run.write_report(tmp_path, run.Report(
        label="B4XPT", workspace="x", cost_usd=258.42, turns=2061, tool_calls=996,
        sessions=4, seconds=17458.7, compactions=1, output_tokens=1_314_077))
    was = run.carried(tmp_path, "B4XPT", resuming=True)
    assert was["cost_usd"] == 258.42
    assert was["turns"] == 2061
    assert was["sessions"] == 4
    assert was["output_tokens"] == 1_314_077
    assert set(was) == set(run.CARRIED)


def test_a_replay_carries_nothing(tmp_path):
    """`--replay` starts the world over, and what it cost to play a world that no
    longer exists is not part of what the new one costs."""
    run.write_report(tmp_path, run.Report(label="B4XPT", workspace="x", cost_usd=258.42))
    assert run.carried(tmp_path, "B4XPT", resuming=False) == dict.fromkeys(run.CARRIED, 0)


def test_the_first_launch_of_a_run_carries_nothing(tmp_path):
    """There is no report to continue from, and asking for one must not be an error."""
    assert run.carried(tmp_path, "B4XPT", resuming=True) == dict.fromkeys(run.CARRIED, 0)


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
    monkeypatch.setattr(run, "session_config", lambda r, label, agent=None: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "max_score", "best_episode", "best_episode_pct",
                        "episodes_completed", "mean_episode", "mean_episode_pct",
                        "score", "score_pct", "lives", "deaths", "unique_cells",
                        "max_level")} | {"episode_scores": [], "achievements": []}))

    def go(stint, max_sessions=3, retries=2):
        starts.clear()
        args = type("A", (), {
            "carry_on": str(tmp_path), "replay": None, "obs": ["pixels"], "budget": 500,
            "stint": stint, "max_sessions": max_sessions, "fresh_world": False,
            "agent": "claude", "model": "m", "dry_run": False, "retries": retries,
            "idle_wait": 0,
        })()
        report = asyncio.run(run.play("W", ("craftax", 0), tmp_path, args, {}, ["W"],
                                      asyncio.Semaphore(1)))
        return report, list(starts)

    report, opened = go(stint=0)
    assert report.sessions == 1 and len(opened) == 1
    # Spelled out rather than left to the actuator's own default, which is now the
    # opposite of what this run asked for. A flag passed only on one side of a
    # default is how the launcher and the daemon come to disagree in silence.
    assert "--same-world" in opened[0], "the world rule rode on a default"

    # `go` carries on the same workspace, so this is a second launch of the run the
    # first one started — which is the whole reason the two numbers below differ.
    # `opened` is one `act init` per session *this launch* played, and `sessions` is
    # every session the *run* has been played by, the first launch's included. This
    # asserted 3 against the second of those, which was right until a run was allowed
    # to outlive a launch and stopped being right without anything failing.
    report, opened = go(stint=10)
    assert len(opened) == 3, "a stinted run stopped chaining"
    assert report.sessions == 4, "the run's session count lost the launch before this"
    assert all("--resume" in one for one in opened), "a chained session restarted the world"
    assert all("--stint" in one for one in opened)


# --------------------------------------------------------------------------- #
# Surviving the night
# --------------------------------------------------------------------------- #


def credential(path: Path, hours: float, days: float = 30.0) -> Path:
    """A stored login whose access token has `hours` left and refresh token `days`."""
    now = time.time() * 1000
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"claudeAiOauth": {
        "accessToken": f"token-{hours}", "refreshToken": "r",
        "expiresAt": int(now + hours * 3_600_000),
        "refreshTokenExpiresAt": int(now + days * 86_400_000),
    }}))
    return path


def test_a_refreshed_credential_is_not_thrown_away(tmp_path, monkeypatch):
    """The failure this exists to prevent, and it only appears on a run long enough
    to need several sessions. An access token lasts about eight hours and nothing
    refreshes the operator's file while a chain is running, so from the second
    session on the session's own refreshed copy is the newer of the two."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    theirs = credential(home / ".claude" / ".credentials.json", hours=8)

    # Session one: nothing local, so it links to the operator's.
    config = run.session_config(tmp_path / "launch", "B4XPT")
    link = config / ".credentials.json"
    assert link.is_symlink() and link.resolve() == theirs.resolve()

    # Session one refreshes: the CLI replaces the link with a copy of a newer token.
    link.unlink()
    credential(link, hours=8)
    # ...and time passes, so the operator's is now the stale one.
    credential(theirs, hours=-0.5)

    run.session_config(tmp_path / "launch", "B4XPT")
    assert not link.is_symlink(), "the refreshed token was replaced by an expired one"
    assert run.credential_life(link)[0] > time.time()


def test_a_stale_local_credential_is_replaced(tmp_path, monkeypatch):
    """The other direction, which is the bug that put the re-seeding here: a copy
    left over from a previous launch expires where the operator's does not."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    theirs = credential(home / ".claude" / ".credentials.json", hours=8)
    config = tmp_path / "launch" / ".sessions" / "B4XPT" / ".claude"
    credential(config / ".credentials.json", hours=-20)

    run.session_config(tmp_path / "launch", "B4XPT")
    link = config / ".credentials.json"
    assert link.is_symlink() and link.resolve() == theirs.resolve()


def test_an_unreadable_credential_is_replaced(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    credential(home / ".claude" / ".credentials.json", hours=8)
    config = tmp_path / "launch" / ".sessions" / "B4XPT" / ".claude"
    config.mkdir(parents=True)
    (config / ".credentials.json").write_text("{ this is not json")

    run.session_config(tmp_path / "launch", "B4XPT")
    assert (config / ".credentials.json").is_symlink()
    assert run.credential_life(tmp_path / "nothing here") == (0.0, 0.0)


def test_a_session_that_plays_nothing_is_retried_and_then_gives_up(tmp_path, monkeypatch):
    """A run meant to go unattended for hours should not die on one API error at
    startup — a session that plays nothing costs almost nothing, so a retry is cheap.
    A chain that cannot make progress must still stop rather than spend the night
    failing, which is what an expired credential looks like from here."""
    ws = tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["pixels"], "actions_used": 0, "terminal": False}))
    plays = iter([10, 0, 0, 10, 0, 0, 0])  # two idle sessions, then progress, then three

    async def fake_act(where, *argv):
        if argv[0] == "init":
            state = json.loads((ws / "state.json").read_text())
            state["actions_used"] += next(plays, 0)
            (ws / "state.json").write_text(json.dumps(state))
        return ""

    monkeypatch.setattr(run, "act", fake_act)
    monkeypatch.setattr(run, "one_session", lambda *a: _nothing())
    monkeypatch.setattr(run, "refresh_brief", lambda w: None)
    monkeypatch.setattr(run, "session_config", lambda r, label, agent=None: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "max_score", "best_episode", "best_episode_pct",
                        "episodes_completed", "mean_episode", "mean_episode_pct",
                        "score", "score_pct", "lives", "deaths", "unique_cells",
                        "max_level")} | {"episode_scores": [], "achievements": []}))

    args = type("A", (), {
        "carry_on": str(tmp_path), "replay": None, "obs": ["pixels"], "budget": 500,
        "stint": 10, "max_sessions": 20, "fresh_world": False, "agent": "claude",
        "model": "m", "dry_run": False, "retries": 2,
        "idle_wait": 0,
    })()
    report = asyncio.run(run.play("W", ("craftax", 0), tmp_path, args, {}, ["W"],
                                  asyncio.Semaphore(1)))
    # 1 plays, 2 and 3 idle, 4 plays (the counter resets), 5-7 idle and it gives up.
    assert report.sessions == 7
    assert run.played(ws) == 20


async def _nothing():
    return "", 0


def test_telemetry_is_banked_across_the_sessions_of_one_run(tmp_path, monkeypatch):
    """An adapter *assigns* cost and token counts, because the CLI reports one
    cumulative figure per session — right within a session, and across a chain it
    keeps only the last one's. The four-session 10,000-action run cost $258 and
    reported $4.64, its last session's figure."""
    ws = tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["pixels"], "actions_used": 0, "terminal": False}))
    per_session = iter([104.57, 86.30, 13.13, 49.77, 4.64])

    async def fake_act(where, *argv):
        if argv[0] == "init":
            state = json.loads((ws / "state.json").read_text())
            state["actions_used"] += 10
            state["terminal"] = state["actions_used"] >= 50
            (ws / "state.json").write_text(json.dumps(state))
        return ""

    async def fake_session(argv, where, env, report, agent):
        # Exactly what Claude.absorb does at the result event: assignment.
        report.cost_usd = next(per_session, 0.0)
        report.output_tokens = 1000
        return "", 0

    monkeypatch.setattr(run, "act", fake_act)
    monkeypatch.setattr(run, "one_session", fake_session)
    monkeypatch.setattr(run, "refresh_brief", lambda w: None)
    monkeypatch.setattr(run, "session_config", lambda r, label, agent=None: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "max_score", "best_episode", "best_episode_pct",
                        "episodes_completed", "mean_episode", "mean_episode_pct",
                        "score", "score_pct", "lives", "deaths", "unique_cells",
                        "max_level")} | {"episode_scores": [], "achievements": []}))

    args = type("A", (), {
        "carry_on": str(tmp_path), "replay": None, "obs": ["pixels"], "budget": 50,
        "stint": 10, "max_sessions": 20, "fresh_world": False, "agent": "claude",
        "model": "m", "dry_run": False, "retries": 2,
        "idle_wait": 0,
    })()
    report = asyncio.run(run.play("W", ("craftax", 0), tmp_path, args, {}, ["W"],
                                  asyncio.Semaphore(1)))
    assert report.sessions == 5
    assert report.cost_usd == pytest.approx(104.57 + 86.30 + 13.13 + 49.77 + 4.64)
    assert report.output_tokens == 5000


def test_a_session_that_never_reported_banks_nothing_twice(tmp_path, monkeypatch):
    """A session killed before its result event leaves the fields as the launcher
    zeroed them, so it contributes nothing — rather than the running total again."""
    ws = tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["pixels"], "actions_used": 0, "terminal": False}))
    reported = iter([10.0, None, 5.0])

    async def fake_act(where, *argv):
        if argv[0] == "init":
            state = json.loads((ws / "state.json").read_text())
            state["actions_used"] += 10
            state["terminal"] = state["actions_used"] >= 30
            (ws / "state.json").write_text(json.dumps(state))
        return ""

    async def fake_session(argv, where, env, report, agent):
        got = next(reported, 0.0)
        if got is not None:      # None = killed before the result event
            report.cost_usd = got
        return "", 0

    monkeypatch.setattr(run, "act", fake_act)
    monkeypatch.setattr(run, "one_session", fake_session)
    monkeypatch.setattr(run, "refresh_brief", lambda w: None)
    monkeypatch.setattr(run, "session_config", lambda r, label, agent=None: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "max_score", "best_episode", "best_episode_pct",
                        "episodes_completed", "mean_episode", "mean_episode_pct",
                        "score", "score_pct", "lives", "deaths", "unique_cells",
                        "max_level")} | {"episode_scores": [], "achievements": []}))

    args = type("A", (), {
        "carry_on": str(tmp_path), "replay": None, "obs": ["pixels"], "budget": 30,
        "stint": 10, "max_sessions": 20, "fresh_world": False, "agent": "claude",
        "model": "m", "dry_run": False, "retries": 2,
        "idle_wait": 0,
    })()
    report = asyncio.run(run.play("W", ("craftax", 0), tmp_path, args, {}, ["W"],
                                  asyncio.Semaphore(1)))
    assert report.cost_usd == pytest.approx(15.0)


# --------------------------------------------------------------------------- #
# The other CLI
# --------------------------------------------------------------------------- #


def test_codex_is_handed_the_same_workspace_and_a_narrower_machine():
    """Two things at once, because they are the same requirement. The brief must be
    the file both arms read, so that a workspace differs between them in nothing; and
    what the session can reach beyond it must be narrowed by name, since this CLI has
    no denylist to pass and ships a browser and agents of its own switched on."""
    codex = AGENTS["codex"]
    argv = codex.argv("PLAY", codex.model, Path("/ws"))
    flat = " ".join(argv)

    assert 'project_doc_fallback_filenames=["CLAUDE.md"]' in argv, "reads AGENTS.md only"
    assert "--ignore-user-config" in argv, "the operator's config would decide the model"
    assert "tools.web_search=false" in argv
    assert "tools.view_image=true" in argv, "the frame channel is what it is measured on"
    for feature in ("browser_use", "computer_use", "multi_agent", "plugins"):
        assert f"--disable {feature}" in flat
    # Its own sandbox cannot start on this machine, and a session whose every command
    # fails answers the prompt anyway rather than stopping. See the class docstring.
    assert "danger-full-access" in argv and "workspace-write" not in argv
    assert argv[-1] == "PLAY", "anything after the prompt is read as part of it"


def test_a_session_gets_the_config_directory_its_own_cli_reads(tmp_path, monkeypatch):
    """The same walk, guarded the same way, for a CLI that spells all three of the
    directory, the credential and the variable differently."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    codex = AGENTS["codex"]
    theirs = home / ".codex" / "auth.json"
    theirs.parent.mkdir(parents=True)
    theirs.write_text(json.dumps({"last_refresh": "2026-09-21T19:43:44.604670599Z"}))

    config = run.session_config(tmp_path / "launch", "B4XPT", codex)
    assert config == tmp_path / "launch" / ".sessions" / "B4XPT" / ".codex"
    link = config / "auth.json"
    assert link.is_symlink() and link.resolve() == theirs.resolve()
    # And the same keep-the-fresher rule, on a stamp that is an age rather than an
    # expiry: a session that refreshed its own copy does not get it taken away.
    link.unlink()
    link.write_text(json.dumps({"last_refresh": "2026-09-22T19:43:44.000000Z"}))
    run.session_config(tmp_path / "launch", "B4XPT", codex)
    assert not link.is_symlink(), "the refreshed token was replaced by an older one"


def history(home: Path, rows: list[tuple[int, dict]]) -> Path:
    """A CLI thread history holding `rows`, as the real one is shaped."""
    home.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(home / AGENTS["codex"].HISTORY)
    db.execute("create table thread_items (created_at_ms int, item_json text)")
    db.executemany("insert into thread_items values (?, ?)",
                   [(at, json.dumps(item)) for at, item in rows])
    db.commit()
    db.close()
    return home


def test_a_frame_opened_by_eye_is_counted_and_audited(tmp_path):
    """The failure this exists to prevent is silent. This CLI's stream carries no
    event for looking at an image, so before harvesting, a session could open every
    observation it had — or the answer key — and the stream would show none of it:
    the frame count read zero and the audit, which is a claim about everything a
    session reached, had never seen the reach."""
    ws = tmp_path / "W"
    ws.mkdir()
    home = history(tmp_path / "home", [
        (1000, {"type": "imageView", "path": "/ws/frames/000001.png"}),
        (1001, {"type": "agentMessage", "text": "not this"}),
    ])
    codex, report = AGENTS["codex"], run.Report(label="W", workspace=str(ws))

    at = run.harvest(codex, home, ws, report, since=0.0)
    assert at == 1000
    assert report.tool_calls == 1, "looking at a frame is a tool call on both arms"

    events = [json.loads(line)
              for line in (ws / "agent_stream.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events] == [HARVESTED], "the message was harvested too"
    assert sum(codex.images(e) for e in events) == 1
    assert codex.ran(events[0]) == ["/ws/frames/000001.png"], "the audit cannot see it"


def test_a_frame_is_not_harvested_twice(tmp_path):
    """A run's sessions share one history and a run outlives the launch that started
    it, so the watermark has to be recoverable from the stream itself: read back as
    zero, a continued run would append every frame the first launch harvested a
    second time and count them all again."""
    ws = tmp_path / "W"
    ws.mkdir()
    home = history(tmp_path / "home", [(1000, {"type": "imageView", "path": "/a.png"})])
    codex, report = AGENTS["codex"], run.Report(label="W", workspace=str(ws))

    run.harvest(codex, home, ws, report, since=0.0)
    assert run.harvested_since(ws) == 1000, "a later launch cannot find the watermark"
    run.harvest(codex, home, ws, report, since=run.harvested_since(ws))

    assert report.tool_calls == 1
    assert len((ws / "agent_stream.jsonl").read_text().splitlines()) == 1


def test_a_shell_that_could_not_start_is_told_apart_from_giving_up():
    """Not the agent's doing, and it does not get better with a retry: when the shell
    cannot start this CLI answers the prompt out of what it expected the commands to
    produce, which leaves a session that spoke and ran nothing."""
    codex = AGENTS["codex"]
    assert codex.no_commands(turns=3, tool_calls=0)
    assert not codex.no_commands(turns=3, tool_calls=9), "it ran things and stopped"
    assert not codex.no_commands(turns=0, tool_calls=0), "it never started"
    assert not AGENTS["claude"].no_commands(turns=3, tool_calls=0)


def test_a_report_that_predates_a_field_is_silent_about_it_not_zero(tmp_path):
    """The bug this exists to prevent cost a published run a session, silently.

    A run outlives a launch, so every launch after the first reads what the run had
    already spent out of the report it is about to rotate. A field that report does
    not carry is a field it is *silent* about — and for `sessions` the difference is
    load-bearing, because a report exists only if a run was played. Read as zero, the
    3,000-action pilot that started the published run vanished: the 30,000-action run
    it grew into reported 12 sessions against the 13 result events in its own stream,
    while its cost — a field that report did carry — came through correctly.
    """
    reports = tmp_path / run.RIG / "reports"
    reports.mkdir(parents=True)
    # Exactly the shape of that pilot's report: a cost, no session count.
    (reports / "W.json").write_text(json.dumps({"cost_usd": 104.57, "turns": 997}))

    was = run.carried(tmp_path, "W", resuming=True)
    assert was["sessions"] == 1, "the session that produced this report was dropped"
    assert was["cost_usd"] == pytest.approx(104.57)
    assert was["compactions"] == 0, "a silent counter is still nought"

    # And a run with nothing behind it has played nothing, which is the other case:
    # no prior run at all is not the same as a prior run that did not say.
    fresh = run.carried(tmp_path, "W", resuming=False)
    assert fresh["sessions"] == 0 and fresh["cost_usd"] == 0


def test_a_turn_is_a_response_and_not_an_event():
    """One API response arrives as several `assistant` events — a thinking block, a
    text block, a tool call — all carrying the same `message.id`. Counted per event,
    a turn stopped meaning a response: the published 30,000-action run reported 7,239
    turns for 3,774 responses.

    What hid it is that `tool_calls` is immune — a `tool_use` block is counted once
    wherever the split puts it — so the report agreed with the read-out on tool calls
    and disagreed on turns, and only one of those two was ever compared.
    """
    claude, report = AGENTS["claude"], run.Report(label="W", workspace="/x")

    def assistant(ident, *blocks):
        return {"type": "assistant", "message": {"id": ident, "content": list(blocks)}}

    thinking = {"type": "thinking", "thinking": "..."}
    text = {"type": "text", "text": "probing left"}
    call = {"type": "tool_use", "name": "Bash", "input": {"command": "./act do left"}}

    for event in (assistant("msg_1", thinking), assistant("msg_1", text),
                  assistant("msg_1", call), assistant("msg_2", call)):
        claude.absorb(report, event)
    assert report.turns == 2, "one response counted more than once"
    assert report.tool_calls == 2, "a tool call is counted wherever the split puts it"

    # An id-less event has nothing to group it by, and folding every one of them under
    # a single absent id would be the opposite mistake.
    claude.absorb(report, assistant(None, text))
    claude.absorb(report, assistant(None, text))
    assert report.turns == 4

    # The tracker is working state, not a finding, and does not reach the report.
    assert "last_message" not in json.loads(report.model_dump_json())


# --------------------------------------------------------------------------- #
# The fence
# --------------------------------------------------------------------------- #


def fenced(argv: list[str], ws: Path) -> subprocess.CompletedProcess:
    """Run `argv` under exactly the fence a codex session of this launch would get.

    The launch's `.bin` is staged first because every real fence stands inside a launch
    that has one, and the allowlist names it: an allowlisted path that does not exist is
    an error rather than something the fence steps over. Without it the fence never
    starts — which a `reads()` assertion cannot tell apart from a file it could not read,
    so the tests below would go green on a fence that was never applied.
    """
    import fence

    if not fence.supported():
        pytest.skip("this kernel has no Landlock, so there is no fence to test")
    run.bin_dir(ws.parent)
    full = run.fenced_argv(AGENTS["codex"], ["codex"], ws, ws)
    wrapped = full[: full.index("--")] + ["--", *argv]
    return subprocess.run(wrapped, cwd=ws, capture_output=True, text=True, timeout=60)


def reads(path: Path, ws: Path) -> bool:
    """Whether a fenced session can read this file."""
    return fenced(["cat", str(path)], ws).returncode == 0


def test_a_fenced_session_cannot_read_what_it_is_meant_to_be_playing_for(tmp_path):
    """The reach this exists to close, named file by file. Each of these was read by
    a real session in its first thirty actions: the package carries every achievement
    by name, `tools/route.py` is a scripted player that walks the early tech tree,
    and the launch's own record says what the run has already scored."""
    ws = tmp_path / "W"
    ws.mkdir()

    assert not reads(ROOT / "tools" / "route.py", ws), "the walkthrough is readable"
    assert not reads(ROOT / "run.py", ws), "the launcher is readable"
    package = run.site_packages(ROOT / ".env-venv") / "craftax"
    assert not reads(package / "craftax" / "constants.py", ws), "the answer is readable"
    # Not a file, because which of the operator's notes exist differs by machine and
    # an assertion that passes because nothing was there proves nothing. The home
    # directory always exists, and a session that cannot list it cannot walk it.
    listed = fenced(["ls", str(Path.home())], ws)
    assert listed.returncode != 0, "the operator's home is walkable"
    # The actuator's own interpreter, which the launch's `.bin` now puts within reach
    # under a name that says nothing. Reaching it is the whole of what 524HW did at its
    # nineteenth tool call; what it got back was ModuleNotFoundError, and this is that.
    blocked = fenced([str(run.bin_dir(ws.parent) / "env-python"), "-c", "import craftax"], ws)
    assert blocked.returncode != 0, "the actuator's interpreter imports the package"


def test_the_fence_keeps_everything_a_session_plays_with(tmp_path):
    """Fenced too tight is a run that cannot play, and the failure would arrive as a
    session that scored nothing rather than as an error. `./act` is the whole of what
    a session does, so what it runs on has to survive: the two files it is made of,
    an interpreter to run them, and a workspace it can write."""
    ws = tmp_path / "W"
    ws.mkdir()

    assert reads(ROOT / "act.py", ws), "the agent's own command is unreadable"
    assert reads(ROOT / "craftax_game.py", ws), "act.py cannot import what it needs"
    # The import machinery lists the directory a script sits in; granted as a listing
    # and not as files, which is why the two above are named one by one.
    listing = fenced(["ls", str(ROOT)], ws)
    assert listing.returncode == 0, "act.py cannot be imported from its own directory"

    done = fenced(["sh", "-c", "echo kept > notes.md && cat notes.md"], ws)
    assert done.stdout.strip() == "kept", "a session cannot keep notes"


def test_the_agents_interpreter_still_opens_an_observation(tmp_path):
    """The frame channel, end to end. A fence that left numpy or Pillow outside would
    take away the one thing the benchmark measures this arm on."""
    ws = tmp_path / "W"
    ws.mkdir()
    done = fenced([str(run.AGENT_PYTHON), "-c",
                   "import numpy, PIL.Image; print('opened')"], ws)
    assert done.stdout.strip() == "opened", done.stderr


def test_only_the_arm_that_reached_for_the_answer_is_fenced():
    """The asymmetry, stated where it can be checked. Fencing the Claude arm would
    change what its finished 30k pass means; leaving this one unfenced would make its
    score a measurement of reading the tech tree."""
    assert AGENTS["codex"].FENCED, "the arm that reached is unfenced"
    assert not AGENTS["claude"].FENCED, "fencing this arm rewrites a finished result"
    blank = run.Report(label="W", workspace="/w")
    assert blank.fenced is False, "a report claims a fence it was not given"


def test_the_wait_between_idle_sessions_doubles_and_is_capped(tmp_path, monkeypatch):
    """Ported from cc_nle, where a chain died inside a rate window: three retries in
    three seconds are three ways of asking the same question inside it. So the wait
    doubles, and stops doubling at an hour. A Codex plan's quota is a week-long
    window — the same shape, sat out with a longer `--retries` rather than a new
    mechanism."""
    ws = tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["pixels"], "actions_used": 0, "terminal": False}))
    waits: list[float] = []

    async def no_wait(seconds):
        waits.append(seconds)

    async def nothing_at_all(where, *argv):
        return ""

    async def nothing(*a):
        return "", 0

    monkeypatch.setattr(run, "act", nothing_at_all)
    monkeypatch.setattr(run, "one_session", nothing)
    monkeypatch.setattr(run, "refresh_brief", lambda w: None)
    monkeypatch.setattr(run, "session_config", lambda r, label, agent=None: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    monkeypatch.setattr(run.asyncio, "sleep", no_wait)
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "max_score", "best_episode", "best_episode_pct",
                        "episodes_completed", "mean_episode", "mean_episode_pct",
                        "score", "score_pct", "lives", "deaths", "unique_cells",
                        "max_level")} | {"episode_scores": [], "achievements": []}))

    def go(retries, idle_wait):
        waits.clear()
        args = type("A", (), {
            "carry_on": str(tmp_path), "replay": None, "obs": ["pixels"], "budget": 500,
            "stint": 10, "max_sessions": 20, "fresh_world": False, "agent": "claude",
            "model": "m", "dry_run": False, "retries": retries, "idle_wait": idle_wait,
        })()
        report = asyncio.run(run.play("W", ("craftax", 0), tmp_path, args, {}, ["W"],
                                      asyncio.Semaphore(1)))
        return report, list(waits)

    _, waited = go(retries=4, idle_wait=60)
    assert waited == [60, 120, 240, 480], "the wait did not double"
    _, waited = go(retries=3, idle_wait=1800)
    assert waited == [1800, run.MAX_IDLE_WAIT, run.MAX_IDLE_WAIT]


def test_a_claude_run_can_be_fenced_and_says_so_before_it_ends(tmp_path, monkeypatch):
    """The published Claude pass stays unfenced; a new Claude model with no record of
    staying in bounds does not have to. `--fence` puts its sessions inside the same
    fence as the Codex arm's, the report says so, and so does the arm record — which
    is written before the first session, because a page built while a run plays has
    no report to read and would otherwise guess from the CLI's default."""
    ws = tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["pixels"], "actions_used": 0, "terminal": False}))
    seen: list[list[str]] = []

    async def fake_act(where, *argv):
        if argv[0] == "init":
            state = json.loads((ws / "state.json").read_text())
            state["actions_used"] += 10
            (ws / "state.json").write_text(json.dumps(state))
        return ""

    async def fake_session(argv, where, env, report, agent):
        arm = json.loads((tmp_path / ".rig" / "arms" / "W.json").read_text())
        assert arm == {"agent": "claude", "model": "claude-fable-5", "fenced": True,
                       "effort": run.AGENTS["claude"].EFFORT}, "no record while it plays"
        seen.append(argv)
        return "", 0

    monkeypatch.setattr(run, "act", fake_act)
    monkeypatch.setattr(run, "one_session", fake_session)
    monkeypatch.setattr(run, "refresh_brief", lambda w: None)
    monkeypatch.setattr(run, "session_config", lambda r, label, agent=None: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    monkeypatch.setattr(run, "fenced_argv", lambda agent, argv, w, h: ["FENCE", *argv])
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "max_score", "best_episode", "best_episode_pct",
                        "episodes_completed", "mean_episode", "mean_episode_pct",
                        "score", "score_pct", "lives", "deaths", "unique_cells",
                        "max_level")} | {"episode_scores": [], "achievements": []}))
    args = type("A", (), {
        "carry_on": str(tmp_path), "replay": None, "obs": ["pixels"], "budget": 500,
        "stint": 0, "max_sessions": 3, "fresh_world": False, "agent": "claude",
        "model": "claude-fable-5", "dry_run": False, "retries": 2, "idle_wait": 0,
        "fence": True, "no_fence": False,
    })()
    report = asyncio.run(run.play("W", ("craftax", 0), tmp_path, args, {}, ["W"],
                                  asyncio.Semaphore(1)))
    assert report.fenced and seen and all(argv[0] == "FENCE" for argv in seen)
    assert seen[0][1] == "claude", "the fence wrapped something other than the CLI"


def test_fence_and_no_fence_cannot_both_be_asked_for():
    done = subprocess.run([sys.executable, str(ROOT / "run.py"), "--fence", "--no-fence",
                           "--dry-run"], capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "not allowed with" in done.stderr
