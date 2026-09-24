#!/usr/bin/env python3
"""One Craftax world, held in this process.

Craftax is an open-ended survival and crafting game: no goal state, no princess,
just a tech tree and a world that kills you. This wraps one instance of it so the
actuator can treat it as "play an action, get an observation" — and holds the three
things the wrapper exists to own.

**The rng schedule.** ``env.step`` is functional and takes a key, so the harness
decides what determinism means. Here it is: the world comes from the seed, and the
key for step *t* is ``fold_in(step_root, t)`` — a function of the position in the
episode and nothing else. So replaying the same actions after a restart replays the
same frames, which is what makes a death cost only the actions already spent
(F4). A key derived from the global action count would look equally deterministic
and quietly destroy that.

**The score.** ``env.step``'s reward is
``Σ(new achievements × coefficient) + 0.1 × Δhealth`` (F1), so it goes negative on
damage and is not the number anything is quoted in. Achievements are read off the
state and diffed per episode; the health term stays in ``reward``, which is what the
agent sees, exactly as an RL policy does.

**The observation channels.** The package renders pixels, text and a symbolic
vector from one state (F6). Which of them a run gives out is configuration, and it
lives here so that ``act.py`` only ever asks for what the run was configured to
give. Adding a channel must never change how the world steps.

They do not cost the same. Measured per action: the jitted 64px render 25 ms, the
symbolic vector 0.2 ms, and ``render_craftax_text`` **381 ms** — it builds a Python
string, so it cannot be jitted and every call re-dispatches its jax work eagerly.
That is nineteen minutes of wall clock on a 3000-action run. It is upstream's
function and stays upstream's: re-implementing it faster would make the text
channel no longer the package's own, which is the only reason it is worth having.

Two upstream facts a caller must not take on trust:

* ``is_terminal`` is true on death, on ``max_timesteps`` **and** on beating the
  boss, and nothing but the state tells them apart (F2). There is no victory to
  read off a reward the way the humanRL games had one.
* Craftax-Classic has no text renderer. ``--obs text`` on it is an error at
  construction rather than a channel that silently arrives empty.
"""

from __future__ import annotations

import contextlib
import functools
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# JAX will otherwise take every core on the box for a 9x11 window, and six of these
# run at once. Set before jax is imported anywhere.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

VARIANTS = ("craftax", "classic")
CHANNELS = ("pixels", "text", "symbolic")
DEFAULT_CHANNELS = ("pixels",)
NOOP = "noop"


class GameError(RuntimeError):
    """Something about the environment is wrong. Never degrade, always raise."""


@dataclass
class Spec:
    """Everything that differs between the two games in the package."""

    env_name: str
    actions: tuple[str, ...]
    achievement_names: tuple[str, ...]
    rewards: Any  # np.ndarray, per-achievement coefficient
    max_score: int
    frame_shape: tuple[int, int, int]
    render: Callable[[Any], Any]
    text: Callable[[Any], str] | None
    symbolic: Callable[[Any], Any]
    levels: bool


@functools.lru_cache(maxsize=None)
def _spec(variant: str) -> Spec:
    """Build the per-variant table. Imported late: jax costs seconds to load.

    Cached because building it loads the texture set, and because the jitted
    functions hung off it are what a second game in the same process would
    otherwise recompile from scratch.

    stdout is swallowed for the same reason `cc_humanrl` swallowed PLE's: the
    package announces "Loading Craftax textures from cache" the first time it draws
    anything, and `./act` is a command whose output the agent reads. Nothing about
    the harness's own housekeeping belongs in that stream.
    """
    import jax  # noqa: F401, PLC0415  — imported for its side effects on XLA_FLAGS
    import numpy as np  # noqa: PLC0415

    def named(achievements) -> tuple[str, ...]:
        """Achievement names in the order the *state vector* uses.

        By `.value`, not by iteration. In Craftax the two disagree: the enum is
        declared with the four tiers grouped together, so iterating it yields
        COLLECT_SAPPHIRE (value 54) at position 25, and 42 of the 67 names land on the
        wrong achievement. The vector `state.achievements` is indexed by value, and so
        is ACHIEVEMENT_REWARD_MAP — which is why the score was right while the names
        were wrong, and why every test passed: the first 25 positions agree, and the
        basic tier is all any test had reached.

        The pilot that finally exposed it reported `enter_graveyard` and
        `defeat_necromancer` at depth 1, which is not a thing that can happen.
        """
        out = [""] * len(achievements)
        for one in achievements:
            out[one.value] = one.name.lower()
        return tuple(out)

    if variant == "craftax":
        from craftax.craftax import constants as C  # noqa: PLC0415
        from craftax.craftax.renderer import (  # noqa: PLC0415
            make_craftax_pixel_renderer,
            render_craftax_symbolic,
            render_craftax_text,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            render = make_craftax_pixel_renderer(C.BLOCK_PIXEL_SIZE_HUMAN)
        return Spec(
            env_name="Craftax-Symbolic-v1",
            actions=tuple(a.name.lower() for a in C.Action),
            achievement_names=named(C.Achievement),
            rewards=np.array(C.ACHIEVEMENT_REWARD_MAP),
            max_score=int(np.array(C.ACHIEVEMENT_REWARD_MAP).sum()),
            frame_shape=(832, 704, 3),
            render=render,
            text=render_craftax_text,
            symbolic=render_craftax_symbolic,
            levels=True,
        )

    from craftax.craftax_classic import constants as C  # noqa: PLC0415
    from craftax.craftax_classic.renderer import (  # noqa: PLC0415
        make_craftax_pixel_renderer,
        render_craftax_symbolic,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        render = make_craftax_pixel_renderer(C.BLOCK_PIXEL_SIZE_HUMAN)
    return Spec(
        env_name="Craftax-Classic-Symbolic-v1",
        actions=tuple(a.name.lower() for a in C.Action),
        # Classic's enum happens to agree with itself, but it is built the same way
        # rather than relying on that.
        achievement_names=named(C.Achievement),
        # Classic has no tiers: one point each, so its denominator is the count.
        rewards=np.ones(len(C.Achievement), dtype=int),
        max_score=len(C.Achievement),
        frame_shape=(576, 576, 3),
        render=render,
        text=None,
        symbolic=render_craftax_symbolic,
        levels=False,
    )


@dataclass
class Engine:
    """The parts of a variant that are stateless, and so are worth compiling once.

    A jitted function is compiled per function object, so a `CraftaxGame` that made
    its own would pay nine seconds for `step` and seven for the renderer every time
    one was constructed. The env objects hold nothing but their static params, so
    one set can back every game of a variant in a process.
    """

    spec: Spec
    env: Any
    params: Any
    step: Any
    reset: Any
    render: Any
    symbolic: Any


@functools.lru_cache(maxsize=None)
def _engine(variant: str) -> Engine:
    import jax  # noqa: PLC0415
    from craftax.craftax_env import make_craftax_env_from_name  # noqa: PLC0415

    spec = _spec(variant)
    # NoAutoReset: a death has to be visible to the harness and the restart has to
    # be the harness's decision, not something that happened inside a step.
    env = make_craftax_env_from_name(spec.env_name, auto_reset=False)
    return Engine(
        spec=spec,
        env=env,
        params=env.default_params,
        step=jax.jit(env.step),
        reset=jax.jit(env.reset),
        render=jax.jit(spec.render),
        symbolic=jax.jit(spec.symbolic),
    )


@dataclass
class Episode:
    """One life, and what it reached before it ended."""

    index: int
    actions: int = 0
    score: int = 0
    reward: float = 0.0
    achievements: list[str] = field(default_factory=list)
    ended: str = ""  # "" | "death" | "timeout" | "boss"

    @property
    def alive(self) -> bool:
        return not self.ended


class CraftaxGame:
    """One world, one budget's worth of observations."""

    def __init__(
        self,
        variant: str = "craftax",
        seed: int = 0,
        obs: tuple[str, ...] = DEFAULT_CHANNELS,
        fresh_world: bool = True,
    ) -> None:
        if variant not in VARIANTS:
            raise GameError(f"{variant!r} is not one of {VARIANTS}")
        channels = tuple(dict.fromkeys(obs))
        if not channels:
            raise GameError("a run with no observation channel is not playable")
        for channel in channels:
            if channel not in CHANNELS:
                raise GameError(f"{channel!r} is not one of {CHANNELS}")

        import jax  # noqa: PLC0415

        self.variant = variant
        self.seed = seed
        self.channels = channels
        self.fresh_world = fresh_world
        engine = _engine(variant)
        self.spec = engine.spec
        if "text" in channels and self.spec.text is None:
            raise GameError(
                f"{variant!r} has no text renderer — the package ships one for "
                "craftax only, so --obs text is not available here"
            )
        self._env = engine.env
        self._params = engine.params
        self._step = engine.step
        self._reset = engine.reset
        self._render = engine.render
        self._symbolic = engine.symbolic

        # Two independent roots, so that the world a seed draws and the dynamics it
        # plays out cannot be confused for one another.
        self._world_root, self._step_root = jax.random.split(jax.random.PRNGKey(seed))

        self.episodes: list[Episode] = []
        self.reward = 0.0  # the raw env reward, summed over the whole run
        self._unlocked: set[str] = set()  # union over every episode
        self._cells: set[tuple[int, int, int]] = set()
        self._max_level = 0
        self._begin()
        self._warm()

    # -- the environment ----------------------------------------------------- #

    def _begin(self) -> None:
        """Generate the world for the next episode and start it."""
        import jax  # noqa: PLC0415

        index = len(self.episodes)
        key = jax.random.fold_in(self._world_root, index if self.fresh_world else 0)
        _, self._state = self._reset(key, self._params)
        self.episodes.append(Episode(index=index))
        self._seen = self._flags()
        self._note_position()

    def _warm(self) -> None:
        """Pay the JIT cost here rather than inside the first batch (F5).

        Compiling `step` is nine seconds and the 64px renderer another seven. A
        session that waits for that on its first `./act do` reads it as a hung
        actuator, and a daemon that pays it per game pays it six times over.
        """
        import jax  # noqa: PLC0415

        key = jax.random.fold_in(self._step_root, 0)
        self._step(key, self._state, 0, self._params)  # discarded
        for channel in self.channels:
            if channel == "pixels":
                self._render(self._state)
            elif channel == "symbolic":
                self._symbolic(self._state)
            elif channel == "text":
                self.spec.text(self._state)

    def play(self, token: str) -> tuple[float, bool]:
        """Play one action. Returns (reward, alive).

        The reward is what `env.step` returns, health term and all — the channel an
        RL policy is given. The score this run is quoted in is read off the
        achievements instead, and never leaves this object.

        A dead world answers nothing: `act.py` stops a batch on the death that
        produced it, so reaching here means a caller ignored that, and stepping a
        terminal state would silently produce frames of a world that has ended.
        """
        import jax  # noqa: PLC0415

        if token not in self.tokens:
            raise GameError(f"{token!r} is not an action of this game")
        episode = self.episodes[-1]
        if not episode.alive:
            return 0.0, False

        key = jax.random.fold_in(self._step_root, episode.actions)
        _, self._state, reward, done, _ = self._step(
            key, self._state, self.tokens.index(token), self._params
        )
        reward = float(reward)
        episode.actions += 1
        episode.reward += reward
        self.reward += reward
        self._collect()
        self._note_position()
        if bool(done):
            episode.ended = self._ending()
        return reward, episode.alive

    def restart(self) -> None:
        """End this life and begin the next.

        By default the next one is a *new world*: `fresh_world=True` rolls the world
        key on by one life, so nothing about this map carries over and a saved
        opening is worth nothing. `fresh_world=False` replays the same one instead —
        the same actions from the start reproduce the same frames, which is what
        lets a session record a good life and re-execute it rather than play.
        """
        self.episodes[-1].ended = self.episodes[-1].ended or "restart"
        self._begin()

    def _ending(self) -> str:
        state = self._state
        if float(state.player_health) <= 0:
            return "death"
        if int(state.timestep) >= int(self._params.max_timesteps):
            return "timeout"
        return "boss"

    @property
    def alive(self) -> bool:
        return self.episodes[-1].alive

    @property
    def tokens(self) -> tuple[str, ...]:
        """The actions, under the names the package gives them."""
        return self.spec.actions

    # -- what the agent is given --------------------------------------------- #

    def frame(self):
        """The screen as an image-shaped (height, width, 3) uint8 array."""
        import numpy as np  # noqa: PLC0415

        return np.asarray(self._render(self._state)).astype(np.uint8)

    def write_frame(self, path: Path) -> Path:
        from PIL import Image  # noqa: PLC0415

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.frame()).save(path, "PNG")
        return path

    def text(self) -> str:
        if self.spec.text is None:
            raise GameError(f"{self.variant!r} has no text renderer")
        return str(self.spec.text(self._state))

    def symbolic(self):
        import numpy as np  # noqa: PLC0415

        return np.asarray(self._symbolic(self._state))

    def observe(self, into: Path, index: int, tag: str = "") -> dict[str, str]:
        """Write every enabled channel, and say where each one went.

        Uniform across channels on purpose: the text render is 1.8KB per action, and
        inlining it would put five megabytes of world into the one file `PROMPT.md`
        tells the session to read back. Paths keep `logs.txt` scannable.

        `tag` names a second observation of the same action — the state a run
        restarted into, written beside the state the action produced rather than
        over it.

        Returns channel -> path, relative to `into`, in a fixed order so a log
        block reads the same way every time. This is the only place that knows
        which channels a run has, which is what keeps `act.py` from growing a
        branch per modality.
        """
        import numpy as np  # noqa: PLC0415

        written: dict[str, str] = {}
        for channel, folder, suffix in (
            ("pixels", "frames", "png"),
            ("text", "text", "txt"),
            ("symbolic", "symbolic", "npy"),
        ):
            if channel not in self.channels:
                continue
            name = f"{folder}/{index:06d}{tag}.{suffix}"
            (into / name).parent.mkdir(parents=True, exist_ok=True)
            if channel == "pixels":
                self.write_frame(into / name)
            elif channel == "text":
                (into / name).write_text(self.text())
            else:
                np.save(into / name, self.symbolic())
            written[channel] = name
        return written

    # -- what the harness keeps to itself ------------------------------------ #
    # The achievement set is the score. None of it is ever written anywhere the
    # agent can read: naming what it has unlocked would hand it the tech tree one
    # rung at a time.

    def state(self):
        """The true world: the block grid, the player's position, the inventory.

        For the scripted route in `tools/route.py`, which is a wiring check played
        harness-side, and for tests. Nothing the agent can reach calls it, and
        nothing derived from it is written into the workspace.
        """
        return self._state

    def _flags(self):
        import numpy as np  # noqa: PLC0415

        return np.asarray(self._state.achievements).astype(bool)

    def _collect(self) -> None:
        """Diff the achievement vector and record anything new (F1, F3)."""
        now = self._flags()
        for i in (now & ~self._seen).nonzero()[0]:
            name = self.spec.achievement_names[int(i)]
            self.episodes[-1].achievements.append(name)
            self._unlocked.add(name)
        self._seen = now
        self.episodes[-1].score = int((now * self.spec.rewards).sum())

    def _note_position(self) -> None:
        state = self._state
        level = int(state.player_level) if self.spec.levels else 0
        x, y = (int(v) for v in state.player_position)
        self._cells.add((level, x, y))
        self._max_level = max(self._max_level, level)

    @property
    def max_score(self) -> int:
        """226 for Craftax, 22 for Classic — the denominator the field quotes."""
        return self.spec.max_score

    def episode_scores(self) -> list[int]:
        """Achievement-weighted return per life, in order. The learning curve."""
        return [e.score for e in self.episodes]

    def achievements_union(self) -> list[str]:
        return sorted(self._unlocked)

    def score_union(self) -> int:
        return int(
            sum(
                self.spec.rewards[self.spec.achievement_names.index(name)]
                for name in self._unlocked
            )
        )

    @property
    def deaths(self) -> int:
        return sum(1 for e in self.episodes if e.ended == "death")

    @property
    def unique_cells(self) -> int:
        return len(self._cells)

    @property
    def max_level(self) -> int:
        return self._max_level
