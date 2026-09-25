"""What the replay page claims about a run, and where it is allowed to be lossy.

The page is a reading of two records that disagree in a useful way: the log is what
the actuator *did*, the stream is what the session *asked for*. Nearly every test here
is about keeping the first as the authority, because the second is the one that lies —
a batch stops early when a life ends, and a session that writes itself a wrapper stops
naming `./act` at all.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from replay import (  # noqa: E402
    BASELINES,
    agent_of,
    assign,
    best_curve,
    attribute,
    batches,
    build,
    keep,
    moments,
    read_log,
    thin,
    trace,
)

import run  # noqa: E402  (replay put rig/ on the path on its way in)
import watch_replay  # noqa: E402

SEP = "=" * 80


def log(*blocks: str) -> str:
    return "".join(f"\n{SEP}\n{b}\n" for b in blocks)


def written(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "logs.txt"
    path.write_text("# preamble, which is not a block\n" + text)
    return path


# --------------------------------------------------------------------------- #
# The log
# --------------------------------------------------------------------------- #


def test_a_block_is_read_the_way_the_actuator_wrote_it(tmp_path):
    path = written(tmp_path, log(
        "action 0 | budget 0/3000 | start\n\n[pixels] frames/000000.png\n",
        "action 1 | budget 1/3000 | left | step 1/2\n\n"
        "plan: face the tree and chop it\n\n[pixels] frames/000001.png\n",
        "action 2 | budget 2/3000 | reward +1 | do | step 2/2\n\n"
        "plan: face the tree and chop it\n\n[pixels] frames/000002.png\n",
    ))
    got = read_log(path)
    assert [b["n"] for b in got] == [0, 1, 2]
    assert got[0]["tok"] == "start" and got[0]["k"] == 0
    assert got[1]["tok"] == "left" and (got[1]["i"], got[1]["k"]) == (1, 2)
    assert got[1]["reward"] == 0.0 and got[1]["plan"] == "face the tree and chop it"
    # `reward` is a field of the header, not the token that was played.
    assert got[2]["tok"] == "do" and got[2]["reward"] == 1.0
    assert got[2]["frames"] == ["frames/000002.png"]


def test_only_the_pixel_channel_is_a_frame(tmp_path):
    """A run may be given text or symbolic observations too, and those are named in
    the same place. The page draws pictures."""
    path = written(tmp_path, log(
        "action 1 | budget 1/3000 | left | step 1/1\n\n"
        "[pixels] frames/000001.png\n[text] text/000001.txt\n"
        "[symbolic] symbolic/000001.npy\n",
    ))
    assert read_log(path)[0]["frames"] == ["frames/000001.png"]


def test_a_life_ending_block_holds_both_observations(tmp_path):
    path = written(tmp_path, log(
        "action 226 | budget 226/3000 | noop | step 5/9\n\n"
        "[pixels] frames/000226.png\n[event] life over: death\n"
        "[pixels] frames/000226-restart.png\n",
    ))
    got = read_log(path)[0]
    assert got["frames"] == ["frames/000226.png", "frames/000226-restart.png"]
    assert got["events"] == ["life over: death"]


# --------------------------------------------------------------------------- #
# Batches
# --------------------------------------------------------------------------- #


def test_a_batch_is_its_step_counter_not_its_command(tmp_path):
    """The pilot wrote itself a `go.py` and played 2568 of its 3066 actions through
    it, so no command in the stream names a batch. `step i/k` says where one begins
    and ends whoever asked for it."""
    path = written(tmp_path, log(
        "action 0 | budget 0/3000 | start\n\n[pixels] frames/000000.png\n",
        "action 1 | budget 1/3000 | left | step 1/2\n\nplan: walk\n",
        "action 2 | budget 2/3000 | do | step 2/2\n\nplan: walk\n",
        "action 3 | budget 3/3000 | up | step 1/1\n\nplan: turn\n",
    ))
    made = batches(read_log(path))
    assert len(made) == 2
    assert made[0]["from"] == 0 and made[0]["to"] == 2
    assert made[0]["toks"] == ["left", "do"] and made[0]["played"] == 2
    assert made[0]["plan"] == "walk"
    assert made[1]["toks"] == ["up"]


def test_a_batch_that_stopped_early_says_how_much_it_asked_for(tmp_path):
    """The world ended the life on action 2 of 9, and the log is the only record that
    seven were dropped — the command still reads `noop*9`."""
    path = written(tmp_path, log(
        "action 1 | budget 1/3000 | noop | step 1/9\n\nplan: wait\n",
        "action 2 | budget 2/3000 | noop | step 2/9\n\nplan: wait\n"
        "[pixels] frames/000002.png\n[event] life over: death\n",
    ))
    made = batches(read_log(path))
    assert made[0]["asked"] == 9 and made[0]["played"] == 2


def test_two_batches_of_the_same_shape_do_not_merge(tmp_path):
    """Both ask for two actions, so only the step counter restarting tells them
    apart — a merge here would hide a plan."""
    path = written(tmp_path, log(
        "action 1 | budget 1/3000 | left | step 1/2\n\nplan: first\n",
        "action 2 | budget 2/3000 | do | step 2/2\n\nplan: first\n",
        "action 3 | budget 3/3000 | right | step 1/2\n\nplan: second\n",
        "action 4 | budget 4/3000 | do | step 2/2\n\nplan: second\n",
    ))
    made = batches(read_log(path))
    assert [b["plan"] for b in made] == ["first", "second"]


# --------------------------------------------------------------------------- #
# The stream, and what belongs to which batch
# --------------------------------------------------------------------------- #


def event(**body) -> str:
    return json.dumps(body)


def assistant(blocks: list[dict], mid: str = "m") -> str:
    return event(type="assistant", message={"id": mid, "content": blocks})


def result_for(tool_id: str, text, error: bool = False) -> str:
    return event(type="user", message={"content": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": text,
         "is_error": error}
    ]})


def test_the_stream_is_flattened_into_what_it_said_ran_and_was_told(tmp_path):
    stream = tmp_path / "s.jsonl"
    stream.write_text("\n".join([
        assistant([{"type": "thinking", "thinking": "", "signature": "x" * 400}]),
        assistant([{"type": "text", "text": "I will chop the tree."}]),
        assistant([{"type": "tool_use", "id": "t1", "name": "Bash",
                    "input": {"command": "./python look.py",
                              "description": "read the frame"}}]),
        result_for("t1", "player at 4,4"),
        assistant([{"type": "tool_use", "id": "t2", "name": "Write",
                    "input": {"file_path": str(tmp_path / "look.py"),
                              "content": "import numpy"}}]),
        result_for("t2", "written"),
    ]))
    got = moments(stream, "claude", tmp_path)
    assert [m["kind"] for m in got] == ["think", "say", "tool", "tool"]
    assert got[0]["sealed"], "an empty thinking block is a sealed one"
    assert got[2]["title"] == "./python look.py" and got[2]["out"] == "player at 4,4"
    assert got[2]["why"] == "read the frame"
    # An absolute path is trimmed in the label; what it wrote stays verbatim.
    assert got[3]["title"] == "./look.py"
    assert got[3]["body"] == "import numpy"


def test_a_tool_result_that_is_an_image_is_counted(tmp_path):
    stream = tmp_path / "s.jsonl"
    stream.write_text("\n".join([
        assistant([{"type": "tool_use", "id": "t1", "name": "Read",
                    "input": {"file_path": "frames/000004.png"}}]),
        result_for("t1", [{"type": "image", "source": {}},
                          {"type": "text", "text": "frames/000004.png"}]),
    ]))
    got = moments(stream, "claude", tmp_path)
    assert got[0]["images"] == 1
    looked, derived = attribute(got)
    assert looked == {"frames/000004.png": 1} and derived == {}


def test_a_crop_is_traced_back_to_the_frame_it_came_from(tmp_path):
    """A session almost never reads a 704x832 frame as it is — it crops a tile first,
    and what enters context is `/tmp/tile.png`. Counting only direct reads would say
    it never looked."""
    stream = tmp_path / "s.jsonl"
    stream.write_text("\n".join([
        assistant([{"type": "tool_use", "id": "t1", "name": "Bash",
                    "input": {"command":
                              "./python -c \"crop('frames/000188.png','/tmp/tile.png')\""}}]),
        result_for("t1", "ok"),
        assistant([{"type": "tool_use", "id": "t2", "name": "Read",
                    "input": {"file_path": "/tmp/tile.png"}}]),
        result_for("t2", [{"type": "image", "source": {}}]),
    ]))
    looked, derived = attribute(moments(stream, "claude", tmp_path))
    assert looked == {}
    assert derived == {"frames/000188.png": 1}


def test_work_belongs_to_the_batch_it_was_preparing(tmp_path):
    """A call played if the budget it reported back went up — which is true of
    `./act do` and equally of a wrapper around it. Everything between two such calls
    is free work, and hangs off the batch that followed it."""
    path = written(tmp_path, log(
        "action 1 | budget 1/3000 | left | step 1/1\n\nplan: one\n",
        "action 2 | budget 2/3000 | do | step 1/1\n\nplan: two\n",
    ))
    made = batches(read_log(path))
    stream = tmp_path / "s.jsonl"
    stream.write_text("\n".join([
        assistant([{"type": "tool_use", "id": "a", "name": "Bash",
                    "input": {"command": "./python look.py"}}]),
        result_for("a", "a tree to the left"),
        assistant([{"type": "tool_use", "id": "b", "name": "Bash",
                    "input": {"command": "./go 'one' left"}}]),
        result_for("b", "ran 1/1: left\nbudget 1/3000 life 1 reward +0 over False"),
        assistant([{"type": "tool_use", "id": "c", "name": "Bash",
                    "input": {"command": "./python world.py"}}]),
        result_for("c", "facing the tree"),
        assistant([{"type": "tool_use", "id": "d", "name": "Bash",
                    "input": {"command": "./go 'two' do"}}]),
        result_for("d", "ran 1/1: do\nbudget 2/3000 life 1 reward +1 over False"),
    ]))
    assign(made, moments(stream, "claude", tmp_path))
    assert [m["title"] for m in made[0]["work"]] == ["./python look.py"]
    assert [m["title"] for m in made[1]["work"]] == ["./python world.py"]
    # And each batch knows the call that played it, whatever that call was named.
    assert made[0]["cmd"] == "./go 'one' left"
    assert made[1]["cmd"] == "./go 'two' do"


def test_one_call_that_played_several_batches_feeds_them_all(tmp_path):
    """A wrapper can loop over `./act do`, so one tool call reports a budget several
    batches ahead of where it started. The work before it belongs to the first of
    them, and every one of them knows what played it."""
    path = written(tmp_path, log(
        "action 1 | budget 1/3000 | left | step 1/1\n\nplan: one\n",
        "action 2 | budget 2/3000 | do | step 1/1\n\nplan: two\n",
    ))
    made = batches(read_log(path))
    stream = tmp_path / "s.jsonl"
    stream.write_text("\n".join([
        assistant([{"type": "tool_use", "id": "a", "name": "Bash",
                    "input": {"command": "./python plan.py"}}]),
        result_for("a", "thinking"),
        assistant([{"type": "tool_use", "id": "b", "name": "Bash",
                    "input": {"command": "./python run.py --steps 2"}}]),
        result_for("b", "budget 1/3000 ...\nbudget 2/3000 life 1 reward +1 over False"),
    ]))
    assign(made, moments(stream, "claude", tmp_path))
    assert [m["title"] for m in made[0]["work"]] == ["./python plan.py"]
    assert made[0]["work"] and not made[1]["work"]
    assert made[0]["cmd"] == made[1]["cmd"] == "./python run.py --steps 2"


# --------------------------------------------------------------------------- #
# Which frames a self-contained page can afford
# --------------------------------------------------------------------------- #


def frame(n, **rest):
    body = {"n": n, "reward": 0.0, "fired": "", "events": [], "kind": "play",
            "i": 2, "k": 9, "dlevel": False}
    return body | rest


def test_thinning_never_drops_a_moment_that_mattered():
    """`--inline` cannot hold 3000 pictures. What it must not do is thin away the
    ones a reader came for."""
    frames = [frame(i) for i in range(300)]
    frames[0]["kind"] = "start"
    frames[40]["reward"] = 3.0
    frames[41]["fired"] = "enter_dungeon"
    frames[42]["dlevel"] = True
    frames[43]["events"] = ["life over: death"]
    frames[44]["kind"] = "restart"
    frames[45]["i"] = 1                      # a batch begins here
    got = keep(frames, budget=20 * 1000, sample=1000)
    assert len(got) < len(frames), "nothing was thinned"
    for i in (0, 40, 41, 42, 43, 44, 45, 299):
        assert i in got, f"frame {i} was thinned away"


def test_thinning_is_a_no_op_when_everything_fits():
    frames = [frame(i) for i in range(50)]
    assert keep(frames, budget=10_000_000, sample=1000) == list(range(50))


def test_thinning_keeps_within_its_budget():
    frames = [frame(i) for i in range(4000)]
    got = keep(frames, budget=8_200_000, sample=6000)
    assert len(got) <= 8_200_000 // 6000 + 2


def test_a_run_of_events_can_exceed_the_budget_and_is_still_kept():
    """Better a page over its budget than one that silently drops the achievements.
    The generator prints the size it wrote, so this is visible rather than quiet."""
    frames = [frame(i, fired="collect_wood") for i in range(100)]
    assert keep(frames, budget=1000, sample=1000) == list(range(100))


def test_the_size_estimate_is_a_mean_not_a_midpoint():
    """A cave frame and a daylit forest frame differ by more than a factor of two, so
    sizing a run off whichever happened to sit at its midpoint is how the first build
    came out at 13.7 MB against a 16 MB limit."""
    frames = [frame(i) for i in range(1000)]
    lean = keep(frames, budget=1_000_000, sample=1_000)
    fat = keep(frames, budget=1_000_000, sample=4_000)
    assert len(lean) > len(fat), "a bigger frame must buy fewer of them"
    assert len(lean) <= 1002 and len(fat) <= 252


# --------------------------------------------------------------------------- #
# Where a run sits
# --------------------------------------------------------------------------- #


def test_the_curve_is_best_life_not_session_total():
    """The quantity the chart plots, and the reason it is that one.

    A new life re-earns every achievement (F12), so a session total counts the same
    unlock once per death. Two runs that reached exactly the same depth would then be
    ranked by how often they died — backwards.
    """
    # Three lives worth 5, 12 and 3, in that order.
    rewards = [5, 12, 3]
    ends = [True, True, True]
    assert best_curve(rewards, ends) == [5, 12, 12]
    assert sum(rewards) == 20, "the session total would have said 20"


def test_the_curve_only_goes_up_and_a_death_does_not_undo_it():
    """Monotonic: what a run reached, it reached. The health term makes a step's
    reward negative without that being a loss of progress."""
    # The life runs 1, 2, 1.9, 2.9, 2.0 — so the best it ever stood at is 2.9, and
    # the health cost at the end takes nothing back.
    curve = best_curve([1, 1, -0.1, 1, -0.9], [False, False, False, False, True])
    assert curve == [1, 2, 2, 2.9, 2.9]
    assert curve == sorted(curve)


def test_a_life_in_progress_counts_before_it_ends():
    """Otherwise a run that never died would plot as a flat zero — which is the two
    expert human runs, both essentially a single life."""
    assert best_curve([2, 3, 4], [False, False, False]) == [2, 5, 9]


def test_thinning_keeps_every_step_where_the_curve_moved():
    """The steps are log-spaced because the runs span 766 to 23,225, and a sampler
    that only took its own spacing would miss the achievements — which are the whole
    shape of the line."""
    curve = [0.0] * 5000
    for i in range(3777, 5000):
        curve[i] = 7.0
    got = thin(curve, points=40)
    steps = [s for s, _ in got]
    assert 3778 in steps, "the one step where anything happened was thinned away"
    assert len(got) < 200 and got[0][0] == 1 and got[-1][0] == 5000
    assert [v for _, v in got] == sorted(v for _, v in got)


def test_thinning_survives_an_empty_run():
    assert thin([]) == []


def test_the_published_baselines_are_the_paper_s_and_are_labelled_by_track():
    """A number here that drifted from the paper would be the quietest possible way
    to make the whole comparison wrong."""
    by_name = {(n, t): p for n, p, t in BASELINES}
    assert by_name[("PPO-GTrXL", "1B")] == 18.3
    assert by_name[("PPO-RNN", "1B")] == 15.3
    assert by_name[("PQN-RNN", "1B")] == 16.0
    assert by_name[("Simulus", "1M")] == 6.6
    assert by_name[("PPO-RNN", "1M")] == 2.3
    assert {t for _, _, t in BASELINES} == {"1B", "1M"}


def test_a_handover_is_a_boundary_and_not_an_action(tmp_path):
    """A run played by several sessions writes a block where each one took over. It
    repeats the action number the run stopped at and its played field is `session N`,
    so left in the stream it becomes a second action 6000 that played "session 3" —
    duplicating the action number and inventing a one-action batch."""
    log = (
        "# preamble\n"
        + SEP + "\n"
        "action 0 | budget 0/10000 | start\n\n[pixels] frames/000000.png\n\n"
        + SEP + "\n"
        "action 6000 | budget 6000/10000 | reward +1 | do | step 12/12\n\n"
        "[pixels] frames/006000.png\n\n"
        + SEP + "\n"
        "action 6000 | budget 6000/10000 | session 3\n\n"
        "[event] resumed: this run was rebuilt from its record at action 6000\n"
        "[pixels] frames/006000-resume.png\n\n"
        + SEP + "\n"
        "action 6001 | budget 6001/10000 | reward +0 | left | step 1/4\n\n"
        "[pixels] frames/006001.png\n\n"
    )
    path = tmp_path / "logs.txt"
    path.write_text(log)
    blocks = read_log(path)

    kinds = [b["kind"] for b in blocks]
    assert kinds.count("handover") == 1
    hand = next(b for b in blocks if b["kind"] == "handover")
    assert hand["n"] == 6000 and hand["session"] == 3

    played = [b for b in blocks if b["kind"] == "action"]
    numbers = [b["n"] for b in played]
    assert numbers == [0, 6000, 6001], numbers
    assert len(numbers) == len(set(numbers)), "the handover duplicated an action number"

    # And it is not a batch: `step` is what says where a batch begins, and it has none.
    made = batches(blocks)
    assert [b["toks"] for b in made] == [["do"], ["left"]]


# --------------------------------------------------------------------------- #
# Watching a run that is still playing
# --------------------------------------------------------------------------- #


def launch(root: Path, label: str, agent: str, played: int, budget: int,
           frames: int | None = None) -> Path:
    """A launch directory holding everything `build` reads and nothing it does not.

    `frames` is how many pictures are actually on disk, and defaults to one per
    action plus the opening observation. Passing fewer sets up the one race a page
    built mid-run has: a log that names a frame the actuator has not finished
    writing.
    """
    from PIL import Image

    ws = root / label
    (ws / "frames").mkdir(parents=True)
    blocks = [f"action 0 | budget 0/{budget} | start\n\n[pixels] frames/000000.png\n"]
    blocks += [
        f"action {n} | budget {n}/{budget} | left | step {n}/{played}\n\n"
        f"plan: walk west\n\n[pixels] frames/{n:06d}.png\n"
        for n in range(1, played + 1)
    ]
    (ws / "logs.txt").write_text("# preamble\n" + log(*blocks))
    for n in range(played + 1 if frames is None else frames):
        Image.new("RGB", (8, 8), (n, n, n)).save(ws / "frames" / f"{n:06d}.png")

    told = f"budget {played}/{budget}"
    stream = [{"type": "thread.started"}, {
        "type": "item.completed",
        "item": {"type": "command_execution", "command": "./act do left",
                 "aggregated_output": told},
    }] if agent == "codex" else [{"type": "system"}, {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                 "input": {"command": "./act do left"}}]},
    }, {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "tool_use_id": "t1",
                                 "content": told}]},
    }]
    (ws / "agent_stream.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in stream))

    (root / run.RIG).mkdir(parents=True, exist_ok=True)
    (root / run.RIG / run.LABELS).write_text(
        json.dumps({label: {"variant": "craftax", "seed": 0}}))
    env = root / ".envs" / label
    env.mkdir(parents=True)
    (env / run.RESULT).write_text(json.dumps({
        "variant": "craftax", "seed": 0, "actions_used": played, "budget": budget,
        "max_score": 226, "max_level": 0, "lives": 1, "deaths": 0,
        "history": ["left"] * played,
    }))
    return root


def page(roots, out: Path) -> dict:
    return build(list(roots), out, inline=False, replay_world=False)


def test_which_cli_played_a_workspace_is_read_off_its_own_stream(tmp_path):
    """A report is written when a session *ends*, so a run watched while it plays has
    none — and reading the agent out of it defaulted to the other CLI. The symptom was
    silent and total: every frame and every action on the page, and not one plan,
    command or line of output, because a Codex stream had been read for a shape it
    does not have."""
    root = launch(tmp_path / "L", "AAAAA", "codex", played=2, budget=10)
    assert not (root / run.RIG / "reports").exists(), "the fixture wrote a report"

    assert agent_of(root / "AAAAA" / "agent_stream.jsonl") == "codex"
    one, = page([root], tmp_path / "p.html")["runs"]
    assert one["agent"] == "codex"
    assert [b["cmd"] for b in one["batches"]] == ["./act do left"], \
        "the stream was read for the wrong CLI and came back empty"


def test_an_absent_stream_is_not_evidence_of_either_cli(tmp_path):
    assert agent_of(tmp_path / "nothing.jsonl") == "claude"


def test_two_launches_are_one_matrix_with_the_arms_told_apart(tmp_path):
    """The comparison the page exists for is one world played by two CLIs, and an arm
    is a launch directory, so it is two of them. Both runs are `craftax:0` — the agent
    is the only thing that names them apart, and the label is the only thing that
    keeps their frames apart."""
    a = launch(tmp_path / "A", "MMMMM", "codex", played=2, budget=10)
    b = launch(tmp_path / "B", "NNNNN", "claude", played=2, budget=10)

    got = page([a, b], tmp_path / "p.html")
    assert got["launch"] == ["A", "B"]
    # Sorted by agent, not by the order the launches were passed in.
    assert [r["agent"] for r in got["runs"]] == ["claude", "codex"]
    assert [r["launch"] for r in got["runs"]] == ["B", "A"]
    assert {r["variant"] for r in got["runs"]} == {"craftax"}
    assert {r["dir"] for r in got["runs"]} == {"p_frames/NNNNN", "p_frames/MMMMM"}


def test_a_run_short_of_its_budget_is_not_offered_as_a_result(tmp_path):
    """A page built on a timer is mostly built over an unfinished run. A matrix that
    showed 2 actions the way it shows 30,000 would be offering a prefix as a result."""
    short = launch(tmp_path / "A", "MMMMM", "codex", played=2, budget=10)
    whole = launch(tmp_path / "B", "NNNNN", "codex", played=10, budget=10)
    assert page([short], tmp_path / "p.html")["runs"][0]["done"] is False
    assert page([whole], tmp_path / "q.html")["runs"][0]["done"] is True


def test_a_frame_is_encoded_once_and_then_kept(tmp_path):
    """The thing that makes a page rebuildable on a timer. At 45 ms a frame, doing all
    thirty thousand again is twenty-three minutes and doing the two hundred that
    arrived since the last build is ten seconds."""
    root = launch(tmp_path / "L", "AAAAA", "claude", played=3, budget=10)
    out = tmp_path / "p.html"
    page([root], out)
    where = out.parent / "p_frames" / "AAAAA"
    was = {f.name: f.stat().st_mtime_ns for f in where.iterdir()}
    assert len(was) == 4, "one per action plus the opening observation"

    page([root], out)
    assert {f.name: f.stat().st_mtime_ns for f in where.iterdir()} == was
    assert not list(where.glob("*.part")), "a temporary file was left behind"


def test_a_world_dealt_again_under_the_same_label_is_encoded_again(tmp_path):
    """`--replay` deals a new world into the same workspace under the same label, and
    writes its frames over the old world's, name for name. Keeping a picture because
    a file of that name exists would show the old run forever."""
    root = launch(tmp_path / "L", "AAAAA", "claude", played=2, budget=10)
    out = tmp_path / "p.html"
    page([root], out)
    kept = out.parent / "p_frames" / "AAAAA" / "000001.webp"
    was = kept.read_bytes()

    from PIL import Image
    fresh = root / "AAAAA" / "frames" / "000001.png"
    Image.new("RGB", (8, 8), (200, 30, 30)).save(fresh)
    os.utime(fresh, (time.time() + 10, time.time() + 10))

    page([root], out)
    assert kept.read_bytes() != was, "the new world is showing the old world's frame"


def test_a_frame_the_log_names_before_the_disk_has_it_waits_for_the_next_build(tmp_path):
    """The log names a frame in the same breath as the actuator writes it, so a page
    built while the world is moving can be one picture ahead of the disk. Everything
    after this point assumes a frame it can open."""
    root = launch(tmp_path / "L", "AAAAA", "codex", played=3, budget=10, frames=3)
    one, = page([root], tmp_path / "p.html")["runs"]
    assert [f["n"] for f in one["frames"]] == [0, 1, 2]


def test_the_world_replay_is_cached_against_the_history_it_came_from(tmp_path,
                                                                     monkeypatch):
    """Three minutes for thirty thousand actions, and a page rebuilt every half hour
    would spend it on a *finished* arm every time it caught the live one up. The key
    is the length of the history rather than an mtime, because a history is only ever
    appended to."""
    import readout

    asked = []

    def counted(record):
        asked.append(len(record["history"]))
        return [{"action": i + 1} for i in range(len(record["history"]))]

    monkeypatch.setattr(readout, "replay", counted)
    root = tmp_path / "L"
    (root / run.RIG).mkdir(parents=True)
    record = {"history": ["left", "right"]}

    assert trace(record, root, "AAAAA") == [{"action": 1}, {"action": 2}]
    assert trace(record, root, "AAAAA") == [{"action": 1}, {"action": 2}]
    assert asked == [2], "a finished arm was replayed a second time"

    record["history"].append("up")
    assert len(trace(record, root, "AAAAA")) == 3
    assert asked == [2, 3], "a run that grew was served the trace it had before"


def test_the_watch_stops_on_the_record_and_not_on_a_process(tmp_path):
    """Not the launcher's pid — a run outlives its launch and this one is continued
    more than once before it reaches its budget — and not `pgrep -f`, whose pattern
    matches the shell that greps for it. The record is what says a run is over."""
    root = launch(tmp_path / "L", "AAAAA", "codex", played=9, budget=10)
    assert not watch_replay.finished([root])

    result = root / ".envs" / "AAAAA" / run.RESULT
    result.write_text(json.dumps(json.loads(result.read_text()) | {"actions_used": 10}))
    assert watch_replay.finished([root])


def test_a_launch_that_has_not_played_yet_is_not_a_launch_that_is_over(tmp_path):
    """The watcher can be started before the run it watches. Reading the absence of a
    record as a finished run would stop the watch before the run began."""
    bare = tmp_path / "M"
    (bare / run.RIG).mkdir(parents=True)
    (bare / run.RIG / run.LABELS).write_text(
        json.dumps({"CCCCC": {"variant": "craftax", "seed": 0}}))
    assert not watch_replay.finished([bare])


def test_a_scan_that_found_nothing_new_looks_the_same_as_the_one_before(tmp_path):
    """What keeps an idle cycle free: a build costs whatever arrived since the last
    one, and overnight nothing does."""
    root = launch(tmp_path / "L", "AAAAA", "codex", played=2, budget=10)
    was = watch_replay.fingerprint([root])
    assert watch_replay.fingerprint([root]) == was

    (root / "AAAAA" / "logs.txt").write_text(
        (root / "AAAAA" / "logs.txt").read_text() + "more\n")
    assert watch_replay.fingerprint([root]) != was


def test_two_runs_of_one_cli_are_told_apart_by_their_model(tmp_path):
    """Four arms is two CLIs twice over, and a row named for its CLI alone would name
    two of them identically. The model is read from the first record that has it: the
    report once a launch is over, and before that what the CLI kept for itself — a
    Codex session's own rollout, a Claude stream's `init` event."""
    sol = launch(tmp_path / "A", "MMMMM", "codex", played=2, budget=10)
    kept = sol / ".sessions" / "MMMMM" / ".codex" / "sessions" / "2026" / "09" / "22"
    kept.mkdir(parents=True)
    (kept / "rollout-x.jsonl").write_text('{"type":"turn_context","payload":{"model":"gpt-5.6-sol"}}\n')

    astra = launch(tmp_path / "B", "NNNNN", "codex", played=2, budget=10)
    (astra / run.RIG / "reports").mkdir(parents=True)
    (astra / run.RIG / "reports" / "NNNNN.json").write_text(
        json.dumps({"model": "gpt-6-astra", "fenced": True}))

    fable = launch(tmp_path / "C", "PPPPP", "claude", played=2, budget=10)
    stream = fable / "PPPPP" / "agent_stream.jsonl"
    stream.write_text(json.dumps({"type": "system", "subtype": "init",
                                  "model": "claude-fable-5"}) + "\n" + stream.read_text())

    got = page([sol, astra, fable], tmp_path / "p.html")["runs"]
    # A run with no stint was one session, so its stint is its budget.
    assert [r["arm"] for r in got] == ["claude fable-5 · stint 10 · same world",
                                       "codex gpt-5.6-sol · stint 10 · same world",
                                       "codex gpt-6-astra · stint 10 · same world"]
    # Fenced by the report where there is one, and by the adapter until then.
    assert [r["fenced"] for r in got] == [False, True, True]


def test_two_runs_of_one_model_are_told_apart_by_stint_and_world(tmp_path):
    """One model on one world, played as one session and as a chain of them, with a
    new world after each death or the same one, is rows with the same CLI and model —
    the stint and the world are the rest of the name."""
    roots = []
    for name, label, stint, fresh in (("A", "SSSSS", 3000, True),
                                      ("B", "TTTTT", 30000, False)):
        root = launch(tmp_path / name, label, "claude", played=2, budget=30000)
        result = root / ".envs" / label / run.RESULT
        result.write_text(json.dumps({**json.loads(result.read_text()),
                                      "stint": stint, "fresh_world": fresh}))
        roots.append(root)
    got = page(roots, tmp_path / "p.html")["runs"]
    assert [r["arm"].split(" · ", 1)[1] for r in got] == ["stint 3k · fresh world",
                                                          "stint 30k · same world"]


def test_the_arm_record_is_read_before_any_report_exists(tmp_path):
    """A fenced Claude run that is still playing has no report, and the CLI's default
    says Claude is unfenced — so without the launcher's own record the page would call
    it unfenced for as long as it played."""
    root = launch(tmp_path / "A", "PPPPP", "claude", played=2, budget=10)
    (root / run.RIG / "arms").mkdir(parents=True)
    (root / run.RIG / "arms" / "PPPPP.json").write_text(json.dumps(
        {"agent": "claude", "model": "claude-fable-5", "fenced": True}))
    one, = page([root], tmp_path / "p.html")["runs"]
    assert one["arm"] == "claude fable-5 · stint 10 · same world" and one["fenced"] is True
