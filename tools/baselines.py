#!/usr/bin/env python3
"""The floor and the ceiling, so that a session's number can be read.

Four policies, all playing by exactly the rules a session plays by — `act.ending`
is shared, so a baseline cannot drift into measuring a different game:

* **random** — uniform over the package's own action space, which is the floor the
  paper's own numbers sit above. `reset` is left out: it is a harness affordance
  rather than a game action, and a policy that resets at random is only spending
  budget more slowly.
* **noop** — the null policy. It should never achieve anything, and it should still
  die: thirst runs out around action 200 whether or not you move, which is the
  cheapest check that the world is actually running.
* **route** — the scripted player in `tools/route.py`, reading the true state and
  walking the early tech tree to iron tools. A ceiling for *competence*, not for the
  game: a person reaches 98% of the maximum and this stops at 6%.
* **replay** — the exact 115 actions `route` produced on seed 0, as a fixed list.
  This is the wiring check, and the analogue of cc_humanrl's 199-action reference
  solution: if replaying it stops producing the same thirteen achievements, the
  environment has changed underneath every other number in the run.

Observations are not written. A baseline needs the score, not the picture, and 30
runs of 3000 actions would otherwise be 90,000 PNGs and half a gigabyte.

    uv run python tools/baselines.py                        # all four
    uv run python tools/baselines.py --policy random --trials 10 --budget 3000
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import route  # noqa: E402
from act import DEFAULT_BUDGET, ending  # noqa: E402
from craftax_game import CraftaxGame  # noqa: E402

POLICIES = ("random", "noop", "route", "replay")

# What `route` played on craftax seed 0, verbatim. Frozen rather than regenerated so
# that a change to the router cannot quietly change the environment's regression
# test into a test of the new router.
REPLAY_SEED = 0
REPLAY = [
    "left", "do", "left", "up", "do", "left", "left", "up", "up", "do", "right",
    "up", "up", "do", "right", "left", "place_table", "make_wood_pickaxe",
    "make_wood_sword", "right", "right", "right", "right", "right", "do",
    "right", "right", "right", "do", "right", "right", "do", "up", "do",
    "down", "do", "left", "up", "do", "up", "up", "do", "left", "left", "left",
    "left", "left", "left", "left", "make_stone_pickaxe", "make_stone_sword",
    "right", "up", "up", "left", "left", "up", "left", "left", "left", "left",
    "left", "left", "left", "do", "left", "down", "do", "right", "down",
    "down", "do", "do", "do", "left", "up", "up", "up", "do", "up", "up",
    "do", "up", "up", "do", "right", "do", "right", "right", "right", "right",
    "right", "right", "right", "up", "do", "right", "up", "do", "left", "up",
    "up", "do", "right", "up", "up", "do", "right", "left", "place_table",
    "down", "up", "place_furnace", "make_iron_pickaxe", "make_iron_sword",
]
# Every basic achievement the route reaches, in the order the engine numbers them.
REPLAY_ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_drink", "make_wood_pickaxe",
    "make_wood_sword", "collect_stone", "make_stone_pickaxe", "make_stone_sword",
    "place_furnace", "collect_coal", "collect_iron", "make_iron_pickaxe",
]


def rollout(variant: str, seed: int, policy: str, budget: int, trial: int) -> dict:
    """One run: a shared budget across lives, ending when the budget does."""
    game = CraftaxGame(variant, seed=seed)
    if policy == "route":
        played = route.run(game, budget)
        used = played.used
    else:
        rng = random.Random(f"{policy}-{variant}-{seed}-{trial}")
        plan = REPLAY if policy == "replay" else None
        used = 0
        while used < budget:
            if plan is not None:
                if used >= len(plan):
                    break
                token = plan[used]
            elif policy == "noop":
                token = "noop"
            else:
                token = rng.choice(game.tokens)
            _, alive = game.play(token)
            used += 1
            # The same rules act.Session plays by: a life ending restarts the world
            # and costs only what was already spent.
            if ending(alive, budget - used) == "restart":
                game.restart()

    best = max(game.episode_scores(), default=0)
    return {
        "variant": variant,
        "seed": seed,
        "policy": policy,
        "trial": trial,
        "actions_used": used,
        "max_score": game.max_score,
        "best_episode": best,
        "best_episode_pct": round(100 * best / game.max_score, 2),
        "score": game.score_union(),
        "score_pct": round(100 * game.score_union() / game.max_score, 2),
        "episode_scores": game.episode_scores(),
        "achievements": game.achievements_union(),
        "lives": len(game.episodes),
        "deaths": game.deaths,
        "unique_cells": game.unique_cells,
        "max_level": game.max_level,
        "alive": game.alive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="baselines", description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="craftax", choices=("craftax", "classic"))
    parser.add_argument("--policy", choices=POLICIES, action="append")
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--trials", type=int, default=5, help="seeds per policy")
    parser.add_argument("--out", type=Path, help="write every row as json")
    args = parser.parse_args()

    budget = args.budget or DEFAULT_BUDGET[args.variant]
    policies = args.policy or list(POLICIES)
    rows = []
    print(f"{'policy':8}{'seed':>5}{'actions':>9}{'best':>6}{'best%':>7}"
          f"{'union':>7}{'union%':>8}{'lives':>7}{'deaths':>8}  achievements")
    for policy in policies:
        seeds = [REPLAY_SEED] if policy == "replay" else range(args.trials)
        for seed in seeds:
            row = rollout(args.variant, seed, policy, budget, trial=seed)
            rows.append(row)
            print(f"{policy:8}{seed:>5}{row['actions_used']:>9}{row['best_episode']:>6}"
                  f"{row['best_episode_pct']:>7.1f}{row['score']:>7}"
                  f"{row['score_pct']:>8.1f}{row['lives']:>7}{row['deaths']:>8}"
                  f"  {len(row['achievements'])}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\n{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
