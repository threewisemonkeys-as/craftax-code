"""What the actuator promises: one action, one observation, and nothing else said.

`cc_humanrl`'s end-to-end check was the 199-action route that wins every game.
Craftax has no route and no win, so what is checked here is the protocol instead: a
batch stops when the state it was planned against is gone, the budget is enforced
before anything reaches the world, a life ending is not the run ending, and nothing
the agent can read names the achievement set.

The one addition to that protocol is the reward, which in the humanRL arm was the
score and so was withheld. Here it is the channel a policy is given, so it is shown
— and what stays hidden is which achievement paid it.
"""

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import act  # noqa: E402
from act import RunState, Session, parse_tokens  # noqa: E402

# Seed 0 opens with a tree to the left: two steps to face it, then chopping.
CHOP = ["left", "left", "do", "do", "do"]
# Standing still is fatal — hunger, thirst and fatigue drain whether or not you act.
# Exactly 226 actions on craftax and 215 on classic, both from seed 0.
STARVE = 400
# The channel for a test that starves and never looks at a frame. Rendering and
# encoding a pixel frame is ~47 ms an action, so a death on pixels costs ten seconds
# or more; on the symbolic vector it costs one.
CHEAP = ("symbolic",)


@pytest.fixture
def rig(tmp_path):
    """A workspace and, somewhere else entirely, the environment's directory."""

    def build(variant="craftax", budget=0, obs=("pixels",), seed=0, fresh_world=False):
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        env_dir = tmp_path / "env"
        env_dir.mkdir(exist_ok=True)
        state = RunState(
            variant=variant,
            seed=seed,
            obs=list(obs),
            fresh_world=fresh_world,
            budget=budget or act.DEFAULT_BUDGET[variant],
            env_dir=str(env_dir),
        )
        return Session.create(ws, state), ws, env_dir

    return build


def run(session, tokens, plan=None):
    args = type("A", (), {"actions": list(tokens), "plan": plan})()
    return act.cmd_do(args, session)


def blocks(ws):
    return (ws / act.LOG).read_text().split(act.SEP)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def test_a_run_is_played_in_batches(rig):
    session, ws, env_dir = rig(budget=40)
    out = run(session, CHOP, plan="chop whatever is to the left")
    assert "ran 5/5" in out
    run(session, ["down"] * 6, plan="walk downhill")

    assert session.state.actions_used == 11
    assert not session.state.terminal, "eleven actions did not end a 40-action run"

    record = json.loads((env_dir / act.RESULT).read_text())
    assert record["max_score"] == 226
    assert "collect_wood" in record["achievements"]
    assert record["score"] > 0 and record["score_pct"] > 0
    assert record["episode_scores"] == [record["best_episode"]]
    assert record["deaths"] == 0
    assert record["unique_cells"] == 7

    log = (ws / act.LOG).read_text()
    assert "plan: chop whatever is to the left" in log
    assert "plan: walk downhill" in log


@pytest.mark.parametrize(
    "obs", [("pixels",), ("text",), ("pixels", "text", "symbolic")]
)
def test_one_observation_per_action_per_channel(rig, obs):
    session, ws, _ = rig(obs=obs)
    run(session, ["noop"] * 3)
    log = (ws / act.LOG).read_text()
    for channel, folder, suffix in (
        ("pixels", "frames", "png"), ("text", "text", "txt"),
        ("symbolic", "symbolic", "npy"),
    ):
        if channel not in obs:
            assert not (ws / folder).exists(), f"{channel} was written but not asked for"
            continue
        written = sorted(p.name for p in (ws / folder).iterdir())
        assert written == [f"{i:06d}.{suffix}" for i in range(4)]
        for name in written:
            assert f"[{channel}] {folder}/{name}" in log


def test_a_death_stops_the_batch_and_the_next_life_begins(rig):
    """A life ending is not the run ending: the budget carries on, in the same
    world, and the block holds both the state it ended in and the state it began
    again in."""
    session, ws, _ = rig()
    out = run(session, ["noop"] * STARVE)

    assert "stopped early: life over: death" in out
    assert "action(s) dropped" in out
    assert session.state.actions_used == 226
    assert session.state.deaths == 1
    assert session.state.lives == 2
    assert not session.state.terminal, "a death is not the end of the run"

    log = (ws / act.LOG).read_text()
    # 226 actions, the state before any of them, the restart — and the preamble,
    # which names the directory too.
    assert log.count("[pixels] frames/") == 229, "the new life got no frame of its own"
    assert log.index("[pixels] frames/000226.png") < log.index("[event] life over: death")
    assert log.index("[event] life over: death") < log.index(
        "[pixels] frames/000226-restart.png"
    )

    start = (ws / "frames" / "000000.png").read_bytes()
    assert (ws / "frames" / "000226.png").read_bytes() != start, "the end is not shown"
    assert (ws / "frames" / "000226-restart.png").read_bytes() == start, (
        "the next life did not begin in the same world"
    )


def test_a_life_ending_costs_only_what_was_spent(rig):
    session, _, _ = rig(budget=500, obs=CHEAP)
    run(session, ["noop"] * STARVE)
    assert session.state.actions_left == 500 - 226
    run(session, CHOP)
    assert session.state.actions_used == 231
    assert session.game.alive


def test_budget_is_refused_before_the_world_is_touched(rig):
    session, ws, _ = rig(budget=10)
    before = (ws / act.LOG).read_text()
    with pytest.raises(SystemExit, match="only 10 left in the budget"):
        run(session, ["noop"] * 11)
    assert session.state.actions_used == 0
    assert (ws / act.LOG).read_text() == before, "a refused batch played something"


def test_budget_ends_the_run(rig):
    session, ws, env_dir = rig(budget=4)
    out = run(session, ["noop"] * 4)
    assert "stopped early: budget spent" in out
    assert session.state.terminal and session.state.outcome == "budget"
    assert "[event] run over: budget" in (ws / act.LOG).read_text()
    assert json.loads((env_dir / act.RESULT).read_text())["score"] == 0

    with pytest.raises(SystemExit, match="this run is over"):
        run(session, ["noop"])


def test_nothing_playable_ends_a_life(rig):
    """The game's 43 actions, and no forty-fourth.

    There was one: `reset`, which abandoned a life and began the next in the same
    world. It cost the first real pilot its run. Because the environment's reward is
    the diff against the *episode's* achievement vector, every life re-earns
    everything, so `reset left do` paid +1 every three actions — and the session found
    it, wrote it down as a decisive finding, and spent 226 resets and 715 lives on it
    while its union score sat at 18. Craftax has no such action; offering one was an
    addition to the interface, and this is the test that there is nothing left to
    find."""
    session, ws, _ = rig()
    run(session, CHOP)
    assert session.playable == session.game.tokens
    assert len(session.playable) == 43
    with pytest.raises(SystemExit, match="not an action"):
        run(session, ["reset"])
    assert session.state.actions_used == 5, "a refused action costs nothing"
    assert session.state.lives == 1
    assert "[event] life over" not in (ws / act.LOG).read_text()


def test_the_curve_across_lives_is_kept_and_the_union_is_not_its_sum(rig):
    """Return per life, in order — the thing no RL baseline has an analogue for — and
    the property that makes the run total worthless as an objective.

    The same two achievements are earned twice, once in each life, and the union still
    says two. A session shown the sum across lives would read that as four, which is
    exactly what the pilot did with 715 of them."""
    session, _, env_dir = rig(budget=600, obs=CHEAP)
    run(session, CHOP)
    run(session, ["noop"] * STARVE)  # stops early: this life is over
    assert session.state.deaths == 1 and session.state.lives == 2
    first = session.state.life_reward
    run(session, CHOP)

    record = json.loads((env_dir / act.RESULT).read_text())
    assert record["episode_scores"] == [2, 2], "the same world paid twice"
    assert record["best_episode"] == 2
    assert record["score"] == 2, "the union is not the sum of the lives"
    assert record["reward"] > 2, "the run total counts the same wood twice"
    # And the total the session is shown starts again with the life.
    assert first == 0.0
    assert session.state.life_reward == pytest.approx(2.0, abs=0.5)


# --------------------------------------------------------------------------- #
# What the workspace is allowed to know
# --------------------------------------------------------------------------- #


def test_state_json_names_neither_the_achievements_nor_the_score(rig):
    """Told that `collect_wood` had fired, a session would know both that wood is a
    thing and that it has some. The tech tree is the game."""
    session, ws, _ = rig()
    run(session, CHOP)
    body = (ws / act.STATE).read_text()
    on_disk = json.loads(body)
    for leak in ("achievements", "score", "max_score", "episodes",
                 "unique_cells", "max_level", "deaths", "env_dir"):
        assert leak not in on_disk, f"state.json exposes {leak}"
    assert "collect_wood" not in body
    assert set(on_disk) >= {"budget", "actions_used", "lives", "life_reward",
                            "terminal"}
    # The run total is kept for analysis and not shown. It is not a score leak; it is
    # worse than that, because summed across lives it counts one achievement many
    # times and a session that optimises it will grind (see `test_nothing_playable...`).
    assert "reward" not in on_disk


def test_the_log_says_nothing_about_the_observation(rig, monkeypatch):
    """No hash, no changed-pixel count, no summary — the path, and that is all.

    The text renderer is stubbed: at 381 ms a call, the 231 actions this needs to
    reach a death cost ninety seconds, and what is checked is the log line that names
    the file, not what the file says (`test_game.py` renders the real thing). The
    second channel is the symbolic one for the same reason: every channel's line is
    written by one function, so the cheapest of them checks it."""
    session, ws, _ = rig(obs=("symbolic", "text"))
    monkeypatch.setattr(session.game, "text", lambda: "a text render")
    run(session, CHOP)
    run(session, ["noop"] * STARVE)  # and through a death, which writes two of each
    for block in blocks(ws)[1:]:
        body = [
            line for line in block.strip().splitlines()
            if line and not line.startswith(("action ", "plan:", "[event]"))
        ]
        assert all(line.startswith(("[symbolic] ", "[text] ")) for line in body), body


def test_the_reward_is_shown_and_what_paid_it_is_not(rig):
    session, ws, _ = rig()
    run(session, CHOP)
    log = (ws / act.LOG).read_text()
    assert "| reward +1" in log, "chopping a tree paid nothing the log admits to"
    assert "collect_wood" not in log
    assert "achievement" not in log.lower()


def test_the_preamble_names_the_actions_the_package_names(rig):
    """Nothing is withheld about the vocabulary here — the point is the game as
    shipped, and `make_wood_pickaxe` is what the package calls it."""
    session, ws, _ = rig()
    assert session.playable == session.game.tokens
    assert len(session.playable) == 43
    preamble = (ws / act.LOG).read_text().split(act.SEP)[0]
    for named in ("noop", "left", "do", "make_wood_pickaxe", "cast_fireball"):
        assert named in preamble
    assert "reset" not in preamble, "the preamble offered an action the game has not"
    assert "226" not in preamble, "the preamble said how much there is to find"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_repeat_syntax():
    playable = ("left", "do", "make_wood_pickaxe", "noop")
    assert parse_tokens(["left", "do*3", "make_wood_pickaxe"], playable) == [
        "left", "do", "do", "do", "make_wood_pickaxe"
    ]
    assert parse_tokens(["make_wood_pickaxe*2"], playable) == ["make_wood_pickaxe"] * 2
    with pytest.raises(SystemExit, match="not an action"):
        parse_tokens(["jump"], playable)
    with pytest.raises(SystemExit, match="not an action"):
        parse_tokens(["k9*2"], playable)
    with pytest.raises(SystemExit, match="repeats 0 times"):
        parse_tokens(["left*0"], playable)
    with pytest.raises(SystemExit, match="no actions given"):
        parse_tokens([], playable)


# --------------------------------------------------------------------------- #
# The daemon, end to end
# --------------------------------------------------------------------------- #


def act_cli(ws: Path, *args: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "act.py"), *args],
        cwd=ws, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_daemon_round_trip(tmp_path):
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    try:
        act_cli(ws, "init", "--variant", "classic", "--env-dir", str(env_dir),
                "--budget", "20")
        assert (ws / "frames" / "000000.png").is_file()
        out = act_cli(ws, "do", "left*3", "--plan", "walk")
        assert "ran 3/3: left left left" in out
        assert "budget 3/20" in act_cli(ws, "status")
        assert "frames/000003.png" in act_cli(ws, "board")
        record = json.loads((env_dir / act.RESULT).read_text())
        assert record["variant"] == "classic" and record["max_score"] == 22
        # The environment's own directory holds the score, and is not anywhere the
        # workspace can be confused for.
        assert not (ws / act.RESULT).exists()

        # A client that hangs up before its reply. The pilot's session was killed
        # mid-batch, `sendall` raised BrokenPipeError out of the accept loop, the
        # daemon exited, `act stop` then failed, and run.py discarded the report of a
        # 2757-action run because its shutdown was untidy.
        rude = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        rude.connect(str(ws / act.SOCKET))
        rude.sendall(json.dumps({"command": "status", "args": {}}).encode() + b"\n")
        rude.close()
        assert "budget 3/20" in act_cli(ws, "status"), "a rude client killed the daemon"
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_again_rotates_the_observations_with_the_log(tmp_path):
    """Two runs interleaved would parse as neither — frames/000012.png must not
    mean two different states."""
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    try:
        act_cli(ws, "init", "--variant", "classic", "--env-dir", str(env_dir),
                "--budget", "20")
        act_cli(ws, "do", "noop*2")
        (ws / "notes.md").write_text("what I worked out")
        act_cli(ws, "init", "--variant", "classic", "--env-dir", str(env_dir),
                "--budget", "20", "--again")
        assert (ws / "logs-attempt1.txt").is_file()
        assert (ws / "frames-attempt1" / "000002.png").is_file()
        assert sorted(p.name for p in (ws / "frames").iterdir()) == ["000000.png"]
        assert (ws / "notes.md").read_text() == "what I worked out", "memory was cleared"
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


# --------------------------------------------------------------------------- #
# One run, several sessions
# --------------------------------------------------------------------------- #
# A run of 30,000 actions is longer than any one agent session, so it is played by
# several. Two things make that a continuation rather than a series of restarts: a
# stint, which ends a session without ending the run, and a rebuild, which recovers
# the world from the record because the world is never saved anywhere.


def stinted(tmp_path, size, budget=20, variant="classic"):
    """A session whose stint is set before the log is written, as `serve` does it."""
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(variant=variant, seed=0, obs=["symbolic"], budget=budget,
                     env_dir=str(env_dir))
    state.begin_stint(size)
    return Session.create(ws, state), ws, env_dir


def test_a_stint_ends_the_session_and_not_the_run(tmp_path):
    session, ws, _ = stinted(tmp_path, 5)
    out = run(session, ["noop*5"])
    assert "stint spent" in out
    assert session.state.stint_over and not session.state.terminal
    assert session.state.actions_left == 15, "a spent stint took the budget with it"
    with pytest.raises(SystemExit, match="your stint is spent"):
        run(session, ["noop"])
    assert "[event] session over" in blocks(ws)[-1]
    assert "this session is over" in act.cmd_status(None, session)


def test_a_batch_larger_than_the_stint_is_refused_before_the_world_is_touched(tmp_path):
    session, _, _ = stinted(tmp_path, 5)
    with pytest.raises(SystemExit, match="only 5 left in your stint"):
        run(session, ["noop*6"])
    assert session.state.actions_used == 0


def test_a_stint_that_runs_out_with_the_budget_is_the_run_ending(tmp_path):
    """`stint_over` is a handover. When the budget goes at the same moment there is
    nobody to hand over to, and saying so would send the launcher looking for a
    session that has nothing to play."""
    session, _, _ = stinted(tmp_path, 5, budget=5)
    out = run(session, ["noop*5"])
    assert session.state.terminal and session.state.outcome == "budget"
    assert not session.state.stint_over
    assert "budget spent" in out


def test_the_preamble_explains_a_stint_only_when_there_is_one(tmp_path):
    _, stint_ws, _ = stinted(tmp_path / "a", 5)
    _, plain_ws, _ = stinted(tmp_path / "b", 0)
    said = (stint_ws / act.LOG).read_text()
    assert "# stint: 5 of those actions are yours" in said
    assert "notes.md" in said, "the session was not told how to talk to its successor"
    assert "# stint:" not in (plain_ws / act.LOG).read_text()


def test_a_rebuilt_run_is_the_same_run(tmp_path):
    """The world is never saved and does not have to be.

    It is a pure function of the seed and the actions taken, so replaying the record
    into a fresh engine must land on the same state. The observation is where that is
    checked rather than the counters: it is derived from the whole of the world —
    the map, the inventory, the position, the time of day — so two that match cannot
    have come from two different states.
    """
    import numpy as np

    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    common = ("--variant", "classic", "--env-dir", str(env_dir), "--obs", "symbolic",
              "--budget", "20", "--stint", "6")
    try:
        act_cli(ws, "init", *common)
        act_cli(ws, "do", "left*2", "do*2", "right", "up")
        assert "this session is over" in act_cli(ws, "status")
        was = json.loads((env_dir / act.RESULT).read_text())

        act_cli(ws, "init", *common, "--resume")
        now = json.loads((env_dir / act.RESULT).read_text())
        assert np.array_equal(
            np.load(ws / "symbolic" / "000006.npy"),
            np.load(ws / "symbolic" / "000006-resume.npy"),
        ), "the rebuilt world is not the world the record was made in"
        for field in ("history", "actions_used", "lives", "reward", "score",
                      "achievements", "episode_scores", "unique_cells", "max_level",
                      "deaths", "best_episode"):
            assert now[field] == was[field], field
        assert now["sessions"] == 2, "the rebuild did not count as a new session"
        assert not now["terminal"] and now["stint_end"] == 12

        # And it carries on from there, into one history rather than two.
        act_cli(ws, "do", "down*2")
        assert json.loads((env_dir / act.RESULT).read_text())["history"] == (
            was["history"] + ["down", "down"]
        )
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_a_rebuilt_run_keeps_what_it_has_already_written(tmp_path):
    """`--again` rotates the log and the observations because the action numbering
    starts over. `--resume` must not: the numbering continues, and rotating would
    hide the run's own first half from the session continuing it."""
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    common = ("--variant", "classic", "--env-dir", str(env_dir), "--obs", "symbolic",
              "--budget", "20")
    try:
        act_cli(ws, "init", *common)
        act_cli(ws, "do", "noop*2")
        before = (ws / act.LOG).read_text()
        act_cli(ws, "init", *common, "--resume")
        after = (ws / act.LOG).read_text()
        assert after.startswith(before), "the log was rewritten rather than continued"
        assert "[event] resumed:" in after
        assert not list(ws.glob("logs-attempt*.txt")) and not list(ws.glob("frames-*"))
        assert (ws / "symbolic" / "000002.npy").is_file()
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_a_rebuilt_run_takes_its_world_from_the_record_not_the_arguments(tmp_path):
    """Variant, seed, channels and fresh_world define the world the history is a
    history of. A launcher that disagreed about any of them would rebuild a
    different world and call it the same run."""
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    try:
        act_cli(ws, "init", "--variant", "classic", "--seed", "0", "--obs", "symbolic",
                "--env-dir", str(env_dir), "--budget", "20")
        act_cli(ws, "do", "noop")
        act_cli(ws, "init", "--variant", "craftax", "--seed", "7", "--obs", "pixels",
                "--env-dir", str(env_dir), "--budget", "20", "--resume")
        now = json.loads((env_dir / act.RESULT).read_text())
        assert now["variant"] == "classic" and now["seed"] == 0
        assert now["obs"] == ["symbolic"] and now["max_score"] == 22
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_resuming_is_refused_when_there_is_nothing_to_resume_or_no_room_to(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    empty = subprocess.run(
        [sys.executable, str(ROOT / "act.py"), "init", "--resume", "--variant", "classic"],
        cwd=ws, capture_output=True, text=True, timeout=120,
    )
    assert empty.returncode != 0 and "no run to resume" in empty.stdout + empty.stderr

    env_dir = tmp_path / "env"
    try:
        act_cli(ws, "init", "--variant", "classic", "--obs", "symbolic",
                "--env-dir", str(env_dir), "--budget", "20")
        act_cli(ws, "do", "noop*4")
        act_cli(ws, "stop")
        small = subprocess.run(
            [sys.executable, str(ROOT / "act.py"), "init", "--resume", "--variant",
             "classic", "--env-dir", str(env_dir), "--budget", "3"],
            cwd=ws, capture_output=True, text=True, timeout=300,
        )
        said = small.stdout + small.stderr
        assert small.returncode != 0 and "at least as large as what is already spent" in said
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_again_and_resume_are_opposites(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    clash = subprocess.run(
        [sys.executable, str(ROOT / "act.py"), "init", "--again", "--resume",
         "--variant", "classic"],
        cwd=ws, capture_output=True, text=True, timeout=120,
    )
    assert clash.returncode != 0
    assert "starts the world over" in clash.stdout + clash.stderr


def test_the_record_is_replaced_and_never_written_in_place(rig, monkeypatch):
    """The launcher builds its whole report out of result.json, and a 30,000-action
    run rewrites it once per action — thirty thousand chances to be killed mid-write.
    A half-written one would lose the run it was recording."""
    session, _, env_dir = rig(variant="classic", budget=10, obs=("symbolic",))
    target = env_dir / act.RESULT
    seen, real = [], Path.write_text

    def spy(self, *a, **kw):
        seen.append(Path(self))
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", spy)
    run(session, ["noop*2"])
    assert seen, "nothing was written at all"
    assert target not in seen, "result.json was written in place"
    assert json.loads(target.read_text())["actions_used"] == 2
    assert not list(env_dir.glob("*.tmp")), "a temporary file was left behind"
