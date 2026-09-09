"""What the wrapper owes the harness.

`cc_humanrl`'s port test was the 199-action reference solution: if it still won, the
wrapper had not broken the game. Craftax has no win to check against, so the
equivalent is the set of invariants everything downstream is built on — that a
prefix replays, that a death is visible and final, that the score is the achievement
set rather than the reward, and that turning an observation channel on does not
change the world it is observing.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from craftax_game import CHANNELS, VARIANTS, CraftaxGame, GameError  # noqa: E402

# Long enough to cut wood, drink, and move off the opening cell; short enough that
# every test that replays it is a second, not a minute.
PREFIX = ["left", "left", "do", "do", "up", "do", "right", "noop", "do", "left"]
# Standing still is fatal: hunger, thirst and fatigue drain whether or not you act.
# 226 actions on craftax and 215 on classic, both from seed 0, both exact.
STARVE = 400


def digest(game: CraftaxGame) -> str:
    return hashlib.sha256(game.frame().tobytes()).hexdigest()


def drive(game: CraftaxGame, plan: list[str]) -> int:
    """Play a plan. Returns actions played, stopping when the world ends."""
    for i, action in enumerate(plan, start=1):
        _, alive = game.play(action)
        if not alive:
            return i
    return len(plan)


# --------------------------------------------------------------------------- #
# The rng schedule
# --------------------------------------------------------------------------- #


def test_a_prefix_replays_bit_exactly_after_a_restart():
    """The invariant the whole harness rests on (F4).

    A death costs only the actions already spent *because* the session can walk
    back to where it was. That is only true if the key for step t is a function of
    t and the seed — a key driven by the global action count would look just as
    deterministic and quietly break this.
    """
    game = CraftaxGame("craftax", seed=0)
    drive(game, PREFIX)
    once, unlocked = digest(game), list(game.episodes[-1].achievements)

    game.restart()
    drive(game, PREFIX)
    assert digest(game) == once, "replaying the same prefix diverged"
    assert game.episodes[-1].achievements == unlocked

    fresh = CraftaxGame("craftax", seed=0)
    drive(fresh, PREFIX)
    assert digest(fresh) == once, "a new game on the same seed diverged"


def test_the_seed_chooses_the_world():
    a, b = CraftaxGame("craftax", seed=0), CraftaxGame("craftax", seed=1)
    assert digest(a) != digest(b)
    assert digest(CraftaxGame("craftax", seed=0)) == digest(a)


def test_a_new_world_is_dealt_only_when_asked():
    """Same world by default: what the session learned about this map still applies,
    which is what makes a second life worth anything."""
    same = CraftaxGame("craftax", seed=0)
    start = digest(same)
    drive(same, PREFIX)
    same.restart()
    assert digest(same) == start

    fresh = CraftaxGame("craftax", seed=0, fresh_world=True)
    assert digest(fresh) == start
    drive(fresh, PREFIX)
    fresh.restart()
    assert digest(fresh) != start


# --------------------------------------------------------------------------- #
# Lives
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", VARIANTS)
def test_standing_still_kills_you_and_the_world_then_says_nothing(variant):
    """There is no goal state here, only ways for a life to end (F2). Upstream will
    happily step a terminal state, which would produce frames of a world that has
    already ended."""
    game = CraftaxGame(variant, seed=0)
    played = drive(game, ["noop"] * STARVE)
    assert played < STARVE, "hunger, thirst and fatigue did not kill a still player"
    assert game.episodes[-1].ended == "death"
    assert not game.alive

    before = digest(game)
    reward, alive = game.play("left")
    assert (reward, alive) == (0.0, False)
    assert digest(game) == before, "the world answered an action after the death"


def test_a_new_life_starts_clean_but_the_run_remembers():
    game = CraftaxGame("craftax", seed=0)
    drive(game, PREFIX)
    won = list(game.episodes[-1].achievements)
    assert won, "the prefix was meant to unlock something"

    game.restart()
    assert game.episodes[-1].achievements == []
    assert game.episodes[-1].score == 0
    assert game.achievements_union() == sorted(won), "the run forgot a life's work"
    assert len(game.episodes) == 2
    assert game.episode_scores()[0] > 0


def test_deaths_are_counted_apart_from_restarts():
    game = CraftaxGame("craftax", seed=0)
    game.restart()
    assert game.deaths == 0, "asking for a new life is not dying"
    drive(game, ["noop"] * STARVE)
    assert game.deaths == 1


# --------------------------------------------------------------------------- #
# The score
# --------------------------------------------------------------------------- #


def test_achievements_are_diffed_off_the_state():
    game = CraftaxGame("craftax", seed=0)
    drive(game, PREFIX)
    won = game.episodes[-1].achievements
    assert "collect_wood" in won
    assert len(won) == len(set(won)), "an achievement was recorded twice"


def test_the_score_is_the_achievement_set_and_the_reward_is_not():
    """`env.step`'s reward carries a 0.1x health term (F1), so it goes negative on
    damage. The agent sees it, exactly as a policy does; nothing is quoted in it."""
    game = CraftaxGame("craftax", seed=0)
    drive(game, ["noop"] * STARVE)
    assert game.episodes[-1].score == 0, "starving to death unlocked something"
    assert game.reward < 0, "nine hearts of damage did not show in the reward"
    assert game.reward == pytest.approx(-0.9, abs=0.05)


def test_the_denominator_is_the_one_the_field_quotes():
    assert CraftaxGame("craftax", seed=0).max_score == 226
    assert CraftaxGame("classic", seed=0).max_score == 22


def test_the_union_is_scored_on_the_same_scale_as_a_life():
    game = CraftaxGame("craftax", seed=0)
    drive(game, PREFIX)
    assert game.score_union() == game.episodes[-1].score
    game.restart()
    assert game.score_union() >= max(game.episode_scores())


def test_exploration_is_read_off_the_engine():
    """Cells are counted from the engine's own player_position, not guessed from the
    actions played — walking into a tree turns you and moves nothing. Seed 0 opens
    with a tree to the left, which is what PREFIX chops; downhill is open."""
    blocked = CraftaxGame("craftax", seed=0)
    drive(blocked, ["left"] * 6)
    assert blocked.unique_cells == 1, "six presses into a tree moved the player"

    walked = CraftaxGame("craftax", seed=0)
    drive(walked, ["down"] * 6)
    assert walked.unique_cells == 7
    assert walked.max_level == 0, "the run started underground"


# --------------------------------------------------------------------------- #
# The observation channels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("variant", "shape", "hud"),
    [("craftax", (832, 704, 3), 576), ("classic", (576, 576, 3), 448)],
)
def test_the_frame_is_the_human_render_and_the_right_way_up(variant, shape, hud):
    """64px/block is what `play_craftax` draws for a person. The inventory panel is
    below the map, so a transposed frame would put it down the side."""
    game = CraftaxGame(variant, seed=0)
    frame = game.frame()
    assert frame.shape == shape and frame.dtype == np.uint8
    assert frame.any(), "the first frame the agent sees is black"
    assert frame[hud:].mean() < frame[:hud].mean(), "the panel is not along the bottom"


@pytest.mark.parametrize(
    "channels",
    [("pixels",), ("text",), ("symbolic",), ("pixels", "text"),
     ("pixels", "text", "symbolic")],
)
def test_every_channel_combination_writes_what_it_claims(channels, tmp_path):
    game = CraftaxGame("craftax", seed=0, obs=channels)
    written = game.observe(tmp_path, 4)
    assert set(written) == set(channels)
    for channel, name in written.items():
        path = tmp_path / name
        assert path.is_file() and path.stat().st_size > 0
        assert name.startswith(f"{channel if channel != 'pixels' else 'frames'}/")
    if "pixels" in channels:
        from PIL import Image

        with Image.open(tmp_path / written["pixels"]) as img:
            assert img.size == (704, 832) and img.mode == "RGB"
    if "text" in channels:
        body = (tmp_path / written["text"]).read_text()
        assert "Map:" in body and "Inventory:" in body
    if "symbolic" in channels:
        assert np.load(tmp_path / written["symbolic"]).shape == (8268,)


def test_the_channels_do_not_change_the_world(tmp_path):
    """Three renderings of one state. If enabling one moved the world, a text run
    and a pixel run would not be comparable, which is the whole reason they share a
    harness."""
    seen = {}
    for i, channels in enumerate([("pixels",), ("text",), ("pixels", "text", "symbolic")]):
        game = CraftaxGame("craftax", seed=0, obs=channels)
        for n, action in enumerate(PREFIX):
            game.play(action)
            game.observe(tmp_path / str(i), n)
        seen[channels] = (digest(game), tuple(game.episodes[-1].achievements))
    assert len(set(seen.values())) == 1, seen


def test_classic_has_no_text_renderer():
    """The package ships one for craftax only, so this is an error at construction
    rather than a channel that arrives empty."""
    with pytest.raises(GameError, match="text renderer"):
        CraftaxGame("classic", seed=0, obs=("pixels", "text"))
    assert CraftaxGame("classic", seed=0, obs=("pixels", "symbolic")).channels


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_the_actions_are_the_packages_own_names():
    game = CraftaxGame("craftax", seed=0)
    assert len(game.tokens) == 43
    assert game.tokens[:6] == ("noop", "left", "right", "up", "down", "do")
    assert "make_wood_pickaxe" in game.tokens
    assert len(CraftaxGame("classic", seed=0).tokens) == 17


@pytest.mark.parametrize("bad", ["jump", "K1", "", "make_gold_pickaxe"])
def test_an_action_this_game_does_not_have_is_refused(bad):
    with pytest.raises(GameError, match="not an action"):
        CraftaxGame("craftax", seed=0).play(bad)


def test_a_run_with_no_channel_or_an_unknown_one_is_refused():
    with pytest.raises(GameError, match="no observation channel"):
        CraftaxGame("craftax", seed=0, obs=())
    with pytest.raises(GameError, match="not one of"):
        CraftaxGame("craftax", seed=0, obs=("pixels", "sound"))
    with pytest.raises(GameError, match="not one of"):
        CraftaxGame("nethack", seed=0)
    assert set(CHANNELS) == {"pixels", "text", "symbolic"}


def test_achievement_names_are_indexed_the_way_the_vector_is():
    """The package's own level table is the oracle, and it is not the enum.

    `state.achievements` is indexed by `Achievement.value`, and so is
    ACHIEVEMENT_REWARD_MAP — but iterating the enum yields a different order, because
    it is declared with the four tiers grouped rather than in value order. Naming by
    iteration put 42 of the 67 names on the wrong achievement while leaving every score
    correct, and left the first 25 positions right, so the whole basic tier agreed and
    nothing caught it until a session reported `enter_graveyard` at depth 1.

    `LEVEL_ACHIEVEMENT_MAP` says which achievement each dungeon level awards, by index.
    Level 1 is the dungeon and level 8 is the graveyard; if the names are aligned it
    reads back as its own documentation, and if they are not it reads as nonsense.
    """
    import numpy as np
    from craftax.craftax import constants as C

    from craftax_game import _spec

    spec = _spec("craftax")
    levels = np.asarray(C.LEVEL_ACHIEVEMENT_MAP)
    assert spec.achievement_names[int(levels[1])] == "enter_dungeon"
    assert spec.achievement_names[int(levels[8])] == "enter_graveyard"
    # And every name is the one whose value is that index.
    for one in C.Achievement:
        assert spec.achievement_names[one.value] == one.name.lower()
    assert len(set(spec.achievement_names)) == len(C.Achievement) == 67

    # The score never depended on this: the reward map is indexed by value too, which
    # is why the pilot's 16.0% survived the bug that mislabelled its achievements.
    assert list(spec.rewards) == [C.achievement_mapping(i) for i in range(67)]


def test_a_deep_achievement_cannot_fire_at_the_surface():
    """What the mislabelling looked like from outside: a session at depth 1 reporting
    the graveyard, which is level 8. Nothing about the tiers should be reachable from
    an unplayed world."""
    game = CraftaxGame(seed=0)
    assert game.achievements_union() == []
    assert game.max_level == 0
