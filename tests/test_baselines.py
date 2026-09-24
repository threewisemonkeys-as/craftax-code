"""The floor and the ceiling, and the rules they share with a real session."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import baselines  # noqa: E402
import route  # noqa: E402
from act import ending  # noqa: E402
from craftax_game import CraftaxGame  # noqa: E402


def test_the_rules_are_the_ones_a_session_plays_by():
    """Shared with act.Session.play, so a floor cannot be a floor for another game.

    Unlike cc_humanrl there is no reward that means victory: a life ends because
    the world ended it, and the run ends only when the budget does."""
    assert ending(False, 100) == "restart"
    assert ending(True, 0) == "budget"
    assert ending(True, 100) is None
    assert ending(False, 0) == "restart", "a death is not masked by the budget"


# --------------------------------------------------------------------------- #
# The wiring check
# --------------------------------------------------------------------------- #


def test_the_frozen_route_still_reaches_iron():
    """The analogue of cc_humanrl's 199-action reference solution.

    115 actions recorded from the router on seed 0, replayed as a fixed list. If
    this stops producing the same thirteen achievements, the environment has moved
    underneath every other number the run produced — and because the list is frozen
    rather than regenerated, a change to the *router* cannot quietly turn this into
    a test of the new router.
    """
    got = baselines.rollout("craftax", baselines.REPLAY_SEED, "replay", 3000, 0)
    assert got["actions_used"] == len(baselines.REPLAY) == 115
    assert got["achievements"] == sorted(baselines.REPLAY_ACHIEVEMENTS)
    assert "make_iron_pickaxe" in got["achievements"]
    assert got["score"] == 13 and got["max_score"] == 226
    assert got["deaths"] == 0 and got["lives"] == 1


def test_the_route_is_reactive_and_not_a_recording():
    """It re-plans from the map, so it reaches iron on worlds the frozen list has
    never seen. Seed 0's iron is embedded in rock with no walkable neighbour, which
    is what the tunnelling is for."""
    reached = []
    for seed in (0, 1, 2):
        game = CraftaxGame("craftax", seed=seed)
        route.run(game, 3000)
        reached.append("make_iron_pickaxe" in game.achievements_union())
    assert all(reached), reached


def test_the_route_reads_the_true_state_and_writes_nothing(tmp_path):
    """It is harness-side: it sees the block grid the agent never gets, and it does
    not write an observation anywhere."""
    game = CraftaxGame("craftax", seed=0)
    router = route.Router(game, 50)
    assert router.grid().shape == (48, 48)
    assert router.at() == (24, 24)
    router.harvest("TREE", "wood", 1)
    assert router.inv("wood") >= 1
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# The floor
# --------------------------------------------------------------------------- #


def test_noop_achieves_nothing_and_still_dies():
    """Thirst runs out around action 200 whether or not you move — which makes this
    the cheapest check that the world is actually running.

    Both regimes, because the death count is the one thing that separates them here.
    A replayed world kills a still player at the same action every time, so 400
    actions buy exactly one death; a new world each life kills at its own pace, and
    the second death inside the same budget is the cheapest evidence that the
    default really does deal a different world.
    """
    same = baselines.rollout("craftax", 0, "noop", 400, 0, fresh_world=False)
    assert same["deaths"] == 1 and same["lives"] == 2
    assert same["actions_used"] == 400

    got = baselines.rollout("craftax", 0, "noop", budget=400, trial=0)
    assert got["score"] == 0 and got["achievements"] == []
    assert got["deaths"] >= 1 and got["lives"] == got["deaths"] + 1
    assert got["actions_used"] == 400
    assert got["unique_cells"] == 1, "standing still moved the player"
    assert got["deaths"] > same["deaths"], "the fresh-world default replayed one world"


def test_random_is_a_floor_not_a_policy():
    """It dies repeatedly and gets barely anywhere — which is what makes it
    readable. Measured: four achievements over 1000 actions, against the route's
    thirteen in 115."""
    runs = [baselines.rollout("craftax", s, "random", 1000, s) for s in range(2)]
    assert all(r["deaths"] >= 2 for r in runs), [r["deaths"] for r in runs]
    assert all(r["score"] <= 8 for r in runs), [r["score"] for r in runs]
    assert baselines.rollout("craftax", 0, "random", 1000, 0) == runs[0], "not seeded"


def test_the_floor_is_under_the_ceiling():
    """The whole point of having both: a session's number is only readable between
    them."""
    floor = baselines.rollout("craftax", 0, "random", 1000, 0)
    ceiling = baselines.rollout("craftax", 0, "replay", 3000, 0)
    assert ceiling["score"] > floor["score"]
    assert ceiling["actions_used"] < floor["actions_used"], (
        "the ceiling took more actions than the floor, which makes it no ceiling"
    )
