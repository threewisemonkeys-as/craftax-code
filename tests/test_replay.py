"""What the replay page claims about a run, and where it is allowed to be lossy.

The page is a reading of two records that disagree in a useful way: the log is what
the actuator *did*, the stream is what the session *asked for*. Nearly every test here
is about keeping the first as the authority, because the second is the one that lies —
a batch stops early when a life ends, and a session that writes itself a wrapper stops
naming `./act` at all.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from replay import (  # noqa: E402
    assign,
    attribute,
    batches,
    keep,
    moments,
    read_log,
)

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
