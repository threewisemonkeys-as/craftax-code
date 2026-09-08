"""What the harness is pinned to.

`cc_humanrl` vendored its games into the repo at a commit, so the thing being played
could not change underneath the results. Craftax is a PyPI package instead, and the
equivalent of that commit is this file: the version is pinned in
tools/make_env_venv.sh and the facts the plan was written against are checked here,
so an upgrade that moves the denominator fails a test rather than quietly restating
every number.

Two of these are load-bearing beyond bookkeeping. The reward map is where the 226
comes from, and 226 is the unit every published baseline is quoted in. And the
functional-JAX determinism check is what makes an action sequence plus a seed the
whole record — if it ever stops holding, frames on disk stop being a cache and the
replay page starts lying.
"""

import subprocess
import sys
from collections import Counter
from pathlib import Path

import jax
import numpy as np
import pytest
from craftax.craftax_env import make_craftax_env_from_name

REPO = Path(__file__).resolve().parents[1]
AGENT_PYTHON = REPO / ".agent-venv" / "bin" / "python"


def test_the_pinned_version_is_the_one_installed():
    import importlib.metadata as meta

    assert meta.version("craftax") == "1.6.1"


def test_the_full_game_is_the_shape_the_plan_assumes():
    from craftax.craftax.constants import (
        ACHIEVEMENT_REWARD_MAP,
        Achievement,
        Action,
        BlockType,
    )

    assert len(Action) == 43
    assert len(Achievement) == 67
    assert len(BlockType) == 37
    tiers = Counter(np.array(ACHIEVEMENT_REWARD_MAP).tolist())
    # 25 basic, 18 intermediate, 15 advanced, 9 very advanced.
    assert tiers == {1: 25, 3: 18, 5: 15, 8: 9}
    assert int(np.array(ACHIEVEMENT_REWARD_MAP).sum()) == 226


def test_classic_pays_one_per_achievement():
    """Classic has no tiers, so its denominator is just the count."""
    from craftax.craftax_classic.constants import Achievement, Action

    assert len(Action) == 17
    assert len(Achievement) == 22


@pytest.mark.parametrize(
    ("name", "shape", "steps"),
    [
        ("Craftax-Symbolic-v1", (832, 704, 3), 100_000),
        ("Craftax-Classic-Symbolic-v1", (576, 576, 3), 10_000),
    ],
)
def test_the_human_render_is_the_frame_we_hand_over(name, shape, steps):
    """64px/block is `play_craftax`'s own setting. The 130x110 RL observation is the
    same information at a size that cannot be read as an image."""
    env = make_craftax_env_from_name(name, auto_reset=False)
    params = env.default_params
    assert params.max_timesteps == steps

    if "Classic" in name:
        from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN as size
        from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
    else:
        from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN as size
        from craftax.craftax.renderer import make_craftax_pixel_renderer

    assert size == 64
    _, state = env.reset(jax.random.PRNGKey(0), params)
    frame = np.array(make_craftax_pixel_renderer(size)(state))
    assert frame.shape == shape
    assert frame.min() >= 0 and frame.max() <= 255
    assert frame.astype(np.uint8).any(), "the first frame the agent would see is black"


def test_a_prefix_replays_bit_exactly():
    """The whole record is the seed and the action sequence, which is only true if
    the per-step key is a function of both and of nothing else."""
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    params = env.default_params
    step = jax.jit(env.step)
    plan = [1, 2, 5, 5, 3, 5, 4, 0, 5, 2, 5, 1]

    def play(seed):
        _, state = env.reset(jax.random.PRNGKey(seed), params)
        trail = []
        for i, action in enumerate(plan):
            _, state, reward, done, _ = step(
                jax.random.fold_in(jax.random.PRNGKey(seed), i), state, action, params
            )
            trail.append((float(reward), bool(done)))
        return state, trail

    once, first = play(7)
    twice, second = play(7)
    assert first == second
    assert np.array_equal(np.array(once.player_position), np.array(twice.player_position))
    assert np.array_equal(np.array(once.map), np.array(twice.map))

    other, _ = play(8)
    assert not np.array_equal(np.array(once.map), np.array(other.map)), (
        "the seed does not choose the world"
    )


def test_an_episode_ends_on_death_or_timeout_and_nothing_else():
    """There is no princess here (F2): `solved` has no analogue, and the score is
    the achievement set."""
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    params = env.default_params
    _, state = env.reset(jax.random.PRNGKey(0), params)
    assert not env.is_terminal(state, params)
    assert env.is_terminal(state.replace(player_health=0.0), params)
    assert env.is_terminal(state.replace(timestep=params.max_timesteps), params)


def test_achievements_are_a_per_episode_flag_vector():
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    params = env.default_params
    _, state = env.reset(jax.random.PRNGKey(0), params)
    unlocked = np.array(state.achievements)
    assert unlocked.shape == (67,) and not unlocked.any()


# --------------------------------------------------------------------------- #
# The fence
# --------------------------------------------------------------------------- #


def test_the_agents_interpreter_can_open_a_png():
    assert AGENT_PYTHON.exists(), "run tools/make_agent_venv.sh"
    subprocess.run(
        [str(AGENT_PYTHON), "-c", "import numpy, PIL"], check=True, capture_output=True
    )


@pytest.mark.parametrize("module", ["craftax", "jax"])
def test_the_agents_interpreter_cannot_reach_the_world_generator(module):
    """`import craftax` is the world generator, the block table, and all 67
    achievements by name. The harness's own interpreter has it; the agent's must
    not, and it is one `sys.executable` slip away from doing so."""
    done = subprocess.run(
        [str(AGENT_PYTHON), "-c", f"import {module}"], capture_output=True
    )
    assert done.returncode != 0, f"the agent's interpreter can import {module}"


def test_the_harnesss_own_interpreter_is_not_the_agents():
    """Both venvs symlink the same base python, so resolving the executable proves
    nothing — what has to differ is the environment it imports from."""
    mine = Path(sys.prefix).resolve()
    theirs = Path(
        subprocess.run(
            [str(AGENT_PYTHON), "-c", "import sys; print(sys.prefix)"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    assert mine != theirs
    assert theirs == (REPO / ".agent-venv").resolve()
