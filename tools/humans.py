#!/usr/bin/env python3
"""Decode the six human trajectories the Craftax README links, and cache them.

    .env-venv/bin/python tools/humans.py ~/craftax-human

The dataset is a Drive link off the README ("run1 is the only trajectory to complete
the game") and is the only record of human play that exists for this environment — the
paper reports no human baseline, unlike Crafter's. Each file is
`{state, action, reward, done}` where `state` is a list of full `EnvState`: the whole
map, position and level, the four condition meters, inventory and the 67-bit
achievement vector, per step.

**Three shims, and the third is a trap.** The pickles predate the package layout, so
they name `craftax.craftax_state`; they predate this jax, so their stored avals carry a
`named_shape` that `ShapedArray.update` no longer accepts. Both are mechanical. The
third is not: **the achievement indices are an older enum**. Decoded against 1.6.1's,
`defeat_necromancer` appears to fire at action 863, before the graveyard is entered. So
this caches only the reward and done channels, which are index-independent, and the
achievement bits are written but must not be scored with today's reward map.

Reading a 23,000-step trajectory costs about a minute and 4 GB, which is why the cache
exists at all: everything downstream reads the npz.
"""

import argparse
import sys
from pathlib import Path


def shim() -> None:
    """Make a 2024 pickle loadable by a 2026 jax and a renamed package."""
    import craftax.craftax.constants as constants  # noqa: PLC0415
    import craftax.craftax.craftax_state as state  # noqa: PLC0415
    from jax._src.core import ShapedArray  # noqa: PLC0415

    sys.modules.setdefault("craftax.craftax_state", state)
    sys.modules.setdefault("craftax.constants", constants)
    update = ShapedArray.update
    ShapedArray.update = lambda self, **kw: update(
        self, **{k: v for k, v in kw.items() if k != "named_shape"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("where", nargs="?", default=str(Path.home() / "craftax-human"),
                        help="a directory of run*.pbz2 (default ~/craftax-human)")
    args = parser.parse_args()

    shim()
    import numpy as np  # noqa: PLC0415
    from craftax.environment_base.util import load_compressed_pickle  # noqa: PLC0415

    root = Path(args.where).expanduser()
    found = sorted(root.glob("run*.pbz2"))
    if not found:
        raise SystemExit(f"humans: no run*.pbz2 under {root}")
    for path in found:
        out = root / f"cache_{path.stem}.npz"
        if out.exists():
            print(f"{path.stem}: cached")
            continue
        got = load_compressed_pickle(str(path))
        done = np.asarray(got["done"]).astype(bool).ravel()
        reward = np.asarray(got["reward"]).astype(float).ravel()
        state = got["state"]
        achievements = (
            np.stack([np.asarray(s.achievements) for s in state]).astype(bool)
            if isinstance(state, (list, tuple))
            else np.asarray(state.achievements).astype(bool)
        )
        np.savez_compressed(out, done=done, reward=reward, ach=achievements)
        print(f"{path.stem}: {len(done)} steps, {int(done.sum())} deaths, "
              f"{reward.sum():.1f} cumulative reward -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
