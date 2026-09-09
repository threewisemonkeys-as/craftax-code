"""What the read-out claims about a run it did not watch.

Two independent things have to hold. The world side must be a reading of *this* run:
it replays the recorded history, so its correspondence to what happened rests on the
replay being faithful — which `tools/baselines.py`'s frozen route pins from outside
this file. And the agent side must count the stream correctly, which is harder than it
looks: one API response arrives as several events sharing a `message.id`, so the
obvious arithmetic counts every request three times and reads a pre-completion stub as
the output.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from baselines import REPLAY, REPLAY_ACHIEVEMENTS, REPLAY_SEED  # noqa: E402
from readout import (  # noqa: E402
    batch_size,
    batches_from_log,
    check,
    curve,
    depth,
    replay,
    session,
    unlocks,
)


def record(history: list[str], **rest) -> dict:
    body = {
        "variant": "craftax", "seed": REPLAY_SEED, "history": history,
        "score": 0, "lives": 1, "deaths": 0,
    }
    return body | rest


# --------------------------------------------------------------------------- #
# The world side
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def route() -> list[dict]:
    return replay(record(REPLAY))


def test_the_replay_reaches_what_the_frozen_route_reached(route):
    """The route's achievements are frozen in `baselines.py`, so this is a check
    against something established elsewhere rather than against itself."""
    fired = [name for row in route for name in row["fired"].split()]
    assert sorted(fired) == sorted(REPLAY_ACHIEVEMENTS)
    # Thirteen achievements and a score of thirteen: everything the route reaches
    # is in the basic tier, which is worth a point each (F3).
    assert route[-1]["score"] == len(REPLAY_ACHIEVEMENTS) == 13
    assert route[-1]["life"] == 1 and route[-1]["deaths"] == 0


def test_the_curve_is_read_at_every_action_not_at_the_end(route):
    """The point of replaying: `result.json` is rewritten in place after each action,
    so the only record of when something fired is the world played again."""
    scores = [row["score"] for row in route]
    assert scores == sorted(scores), "a union score cannot go down"
    assert scores[0] == 0 and scores[-1] == 13
    # Wood at action 2 is the first thing on this route, and the frozen list starts
    # there, so an off-by-one in the numbering would show up here.
    first = next(row for row in route if row["fired"])
    assert first["action"] == 2 and first["fired"] == "collect_wood"


def test_a_death_restarts_the_world_the_way_the_actuator_does():
    """The restart after a death is the actuator's, so it is not in the history and
    has to be reproduced rather than read. Seed 0 starves at exactly 226 noops
    (tests/test_act.py), which is what makes this an outside expectation."""
    rows = replay(record(["noop"] * 230, lives=2, deaths=1))
    died = [row for row in rows if not row["alive"]]
    assert [row["action"] for row in died] == [226]
    assert rows[-1]["life"] == 2 and rows[-1]["deaths"] == 1
    # The new life is a life, not a continuation: its own score starts at nothing.
    assert rows[-1]["life_score"] == 0
    # And the resources run down before it, which is what killed it.
    assert rows[224]["drink"] == 0 or rows[224]["food"] == 0


def test_a_reset_token_restarts_without_stepping():
    """`reset` is the actuator's, not the environment's: it spends an action and ends
    the life, and the world it lands in is not one the action moved."""
    rows = replay(record(["noop", "reset", "noop"]))
    assert [row["life"] for row in rows] == [1, 2, 2]
    assert rows[1]["reward"] == 0.0


def test_the_check_says_when_the_replay_is_not_this_run(route):
    assert "matches" in check(route, record(REPLAY, score=13, lives=1, deaths=0))
    assert "DIVERGED" in check(route, record(REPLAY, score=14, lives=1, deaths=0))


def test_the_curve_and_the_unlocks_are_tables(route):
    rows = curve(route, len(REPLAY), step=50)
    assert rows[0][0] == "action"
    assert [row[0] for row in rows[1:]] == [50, 100, 150]
    assert rows[-1][1] == 13
    assert len(unlocks(route)) == 1 + len(REPLAY_ACHIEVEMENTS)


# --------------------------------------------------------------------------- #
# The agent side
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("command", "actions"),
    [
        ("./act do left", 1),
        ("./act do left do*5", 6),
        ("./act do down*12 --plan 'walk down and see if anything moves'", 12),
        # The plan is prose and can say anything, including something with a star in
        # it. Counting it would make the batch size a function of the briefing.
        ("./act do noop --plan 'wait 10*10 and watch'", 1),
        ("./act status", 0),
        ("./python look.py", 0),
        # A compound command plays both batches, and the shell separator ends the
        # first one — counting `left;` and `./act` as actions is how a naive scan
        # turns two actions into four.
        ("./act do left; ./act do right", 2),
        ("./act do left && ./python look.py", 1),
    ],
)
def test_a_batch_is_what_the_session_asked_for(command, actions):
    assert batch_size(command) == actions


def assistant(mid: str, blocks: list[dict], usage: dict) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {"id": mid, "content": blocks, "usage": usage},
        "timestamp": "2026-09-09T00:00:00.000Z",
    })


def test_what_a_turn_wrote_is_read_from_text_and_commands(tmp_path):
    """Thinking reaches the stream as a signature and no text — 119 blocks of it in the
    pilot, 422KB of signature, zero characters of thought. So the only measurable
    output is the visible text and the commands, and a turn's share of the total is
    taken from those."""
    stream = tmp_path / "agent_stream.jsonl"
    stream.write_text("\n".join([
        assistant("m1", [
            {"type": "thinking", "thinking": "", "signature": "s" * 3552},
            {"type": "text", "text": "y" * 20},
            {"type": "tool_use", "name": "Bash", "input": {"command": "./act do left"}},
        ], {"cache_read_input_tokens": 10}),
        json.dumps({"type": "result", "usage": {"output_tokens": 900}}),
    ]))
    got = session(stream)
    row = got["turn_rows"][0]
    assert row["chars"] > 20, "the command it issued is output it wrote"
    assert row["chars"] < 200, "the signature is not output it wrote"
    assert row["output"] == 900


def test_one_response_is_one_turn_however_many_events_it_arrives_as(tmp_path):
    """Three events, one `message.id`, one request's worth of tokens. Summing the
    events would report three turns and triple the context they were charged for."""
    usage = {"input_tokens": 2, "cache_read_input_tokens": 11439,
             "cache_creation_input_tokens": 1335, "output_tokens": 3}
    stream = tmp_path / "agent_stream.jsonl"
    stream.write_text("\n".join([
        assistant("m1", [{"type": "thinking", "thinking": "x" * 100}], usage),
        assistant("m1", [{"type": "text", "text": "y" * 10}], usage),
        assistant("m1", [{"type": "tool_use", "name": "Bash",
                          "input": {"command": "./act do left do*5"}}], usage),
        json.dumps({"type": "result", "usage": {"output_tokens": 4000},
                    "total_cost_usd": 1.5, "num_turns": 1,
                    "modelUsage": {"claude-opus-5": {"thinkingTokens": 3000}}}),
    ]))
    got = session(stream)
    assert got["turns"] == 1
    assert got["tool_calls"] == 1
    assert got["turn_rows"][0]["cache_read"] == 11439
    assert got["batches"] == [6]
    assert got["kinds"] == {"act do": 1} and got["tools"] == {"Bash": 1}
    # The stub says 3; the session's own total says 4000, and one turn gets all of it.
    assert got["output_total"] == 4000
    assert got["turn_rows"][0]["output"] == 4000
    assert got["thinking_total"] == 3000 and got["cost"] == 1.5


def test_frames_and_errors_and_the_window_are_read(tmp_path):
    stream = tmp_path / "agent_stream.jsonl"
    stream.write_text("\n".join([
        assistant("m1", [{"type": "tool_use", "name": "Bash",
                          "input": {"command": "./act board"}}],
                  {"cache_read_input_tokens": 10}),
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "content": [
                {"type": "image", "source": {}},
                {"type": "text", "text": "frames/000004.png"},
            ]},
            {"type": "tool_result", "content": "no such file", "is_error": True},
        ]}}),
        json.dumps({"type": "system", "subtype": "compact_boundary"}),
        json.dumps({"type": "rate_limit_event", "rate_limit_info": {
            "status": "allowed",
            "unifiedWindows": {"five_hour": {"utilization": 0.42},
                               "seven_day": {"utilization": 0.07}},
        }}),
    ]))
    got = session(stream)
    assert got["frames"] == 1 and got["errors"] == 1 and got["compactions"] == 1
    assert got["five_hour"] == 0.42 and got["seven_day"] == 0.07
    assert got["limit_status"] == "allowed" and got["limits"] == 1
    assert got["kinds"] == {"act other": 1}


def test_the_depth_table_apportions_the_measured_total(tmp_path):
    """Per-turn cost is not reported, so the table is the session's own total split by
    weighted tokens. What it must not do is invent a total."""
    stream = tmp_path / "agent_stream.jsonl"
    lines = [
        assistant(f"m{n}", [{"type": "text", "text": "z" * 10},
                            {"type": "tool_use", "name": "Bash",
                             "input": {"command": "./act do left"}}],
                  {"input_tokens": 1, "cache_read_input_tokens": 1000 * (n + 1),
                   "cache_creation_input_tokens": 0, "output_tokens": 3})
        for n in range(10)
    ]
    lines.append(json.dumps({"type": "result", "usage": {"output_tokens": 100},
                             "total_cost_usd": 2.0, "duration_ms": 60000}))
    stream.write_text("\n".join(lines))
    got = session(stream)
    rows = depth(got, bands=5)
    assert rows[0][0] == "turns" and len(rows) == 6
    apportioned = sum(float(row[-1]) for row in rows[1:])
    assert apportioned == pytest.approx(2.0, abs=0.02)
    # Later bands read more context, so they carry more of the bill.
    assert float(rows[1][-1]) < float(rows[-1][-1])


def test_a_missing_stream_is_not_a_crash(tmp_path):
    got = session(tmp_path / "nothing.jsonl")
    assert got["turns"] == 0 and got["cost"] == 0.0 and depth(got) == []


def test_the_log_knows_the_batches_even_when_the_stream_does_not(tmp_path):
    """The pilot session wrote itself a `./go` wrapper on top of `./act do`, after
    which no command in the stream names a batch. The actuator writes `step 1/n` into
    every block it logs, so the sizes survive the wrapper."""
    (tmp_path / "logs.txt").write_text(
        "action 1 | budget 1/3000 | left | step 1/3\n"
        "action 2 | budget 2/3000 | reward +1 | do | step 2/3\n"
        "action 3 | budget 3/3000 | left | step 3/3\n"
        "action 4 | budget 4/3000 | noop | step 1/1\n"
    )
    (tmp_path / "logs-attempt1.txt").write_text(
        "action 1 | budget 1/3000 | up | step 1/12\n"
    )
    assert sorted(batches_from_log(tmp_path)) == [1, 3, 12]
    assert batches_from_log(tmp_path / "nowhere") == []
