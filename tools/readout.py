#!/usr/bin/env python3
"""What a finished session actually did, world side and agent side.

Two questions M5 exists to answer, and neither is in the report the launcher writes.

**Where does the achievement curve flatten?** The report carries the score at the end
and the score per life, which cannot say whether a life spent its last thousand
actions learning anything. The answer needs the score *at every action*, and nothing
records it: ``result.json`` is rewritten in place after each one. So this replays the
run — the token history is in the record and a world is bit-exact in its seed and its
actions (F5) — and reads the score off the engine as it goes. The replay's final score
is checked against the record, which is what makes it a reading of this run rather
than of a similar one.

**What did each action cost?** Actions per tool call, tokens per turn as the context
grows, and where the money went — all from the event stream, which carries per-turn
usage and, on a plan login, the five-hour window it is spending.

    .env-venv/bin/python tools/readout.py ~/craftax-runs/20260909-043025

Cost per band is the reported total *apportioned* by weighted tokens, not a
measurement: the CLI reports one figure for the session, and the weights (cache read
0.1x input, cache write 1.25x, output 5x) are the published ratio structure rather
than something this run establishes.
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from act import RESET  # noqa: E402

RIG = ".rig"
# The published ratio structure, not a price: only the shape matters, because the
# total it is apportioning is measured.
WEIGHTS = {"input": 1.0, "cache_write": 1.25, "cache_read": 0.1, "output": 5.0}


# --------------------------------------------------------------------------- #
# The world side
# --------------------------------------------------------------------------- #


def replay(record: dict) -> list[dict]:
    """Play the recorded history again and read the score off the engine each action.

    Mirrors ``act.Session.play``'s control flow exactly, which is the whole of the
    correspondence: a life that ends restarts the world, and that restart is the
    actuator's, so it is not in the history and has to be reproduced here rather than
    read.

    `reset` is handled for the same reason. It is no longer an action — it was removed
    after the first pilot ground it — but the runs that recorded it are still runs, and
    a tool that could not read them would make that pilot's history unreadable.
    """
    from craftax_game import CraftaxGame  # noqa: PLC0415

    game = CraftaxGame(
        variant=record["variant"],
        seed=record["seed"],
        obs=("symbolic",),  # never written; the cheapest channel to not use
        fresh_world=record.get("fresh_world", False),
    )
    rows, seen = [], set()
    for i, token in enumerate(record["history"], start=1):
        if token == RESET:
            game.restart()
            reward, alive = 0.0, True
        else:
            reward, alive = game.play(token)
        union = set(game.achievements_union())
        fired = sorted(union - seen)
        seen = union
        state = game.state()
        rows.append({
            "action": i,
            "token": token,
            "reward": round(reward, 3),
            "score": game.score_union(),
            "fired": " ".join(fired),
            "life": len(game.episodes),
            "life_score": game.episodes[-1].score,
            "level": int(getattr(state, "player_level", 0)),
            "health": round(float(state.player_health), 2),
            "food": int(state.player_food),
            "drink": int(state.player_drink),
            "energy": int(state.player_energy),
            "alive": int(alive),
            "deaths": game.deaths,
        })
        if not alive:
            game.restart()
    return rows


def check(rows: list[dict], record: dict) -> str:
    """Did the replay land where the run did?"""
    if not rows:
        return "no history to replay"
    was, now = record["score"], rows[-1]["score"]
    lives, deaths = rows[-1]["life"], rows[-1]["deaths"]
    ok = (was == now and lives == record["lives"] and deaths == record["deaths"])
    return (
        f"replay {'matches' if ok else 'DIVERGED FROM'} the record: "
        f"score {now} vs {was}, lives {lives} vs {record['lives']}, "
        f"deaths {deaths} vs {record['deaths']}"
    )


def curve(rows: list[dict], budget: int, step: int = 250) -> list[list]:
    out = [["action", "union", "%", "life", "life score", "deaths", "level",
            "food", "drink", "energy"]]
    for edge in range(step, budget + step, step):
        got = [r for r in rows if r["action"] <= edge]
        if not got:
            break
        r = got[-1]
        out.append([edge, r["score"], f"{100 * r['score'] / 226:.1f}", r["life"],
                    r["life_score"], r["deaths"], r["level"], r["food"], r["drink"],
                    r["energy"]])
    return out


def unlocks(rows: list[dict]) -> list[list]:
    out = [["action", "life", "achievement"]]
    for r in rows:
        for name in r["fired"].split():
            out.append([r["action"], r["life"], name])
    return out


# --------------------------------------------------------------------------- #
# The agent side
# --------------------------------------------------------------------------- #

# One `act do` call, up to whatever ends it: the shell's separators, a newline, or
# the plan, which is prose and may contain anything at all.
BATCH = re.compile(r"\./act\s+do\s+([^\n;&|]*)")
# What act.py's own parser would take: a name, optionally repeated.
TOKEN = re.compile(r"^[a-z_]+(?:\*(\d+))?$")
KIND = (
    ("act do", re.compile(r"\./act\s+do\b")),
    ("act other", re.compile(r"\./act\s+(?!do\b)")),
    ("python", re.compile(r"(?:\./python|python3?)\b")),
)


def batch_size(command: str) -> int:
    """How many actions a command's `./act do` calls asked for.

    What is wanted is what the session *asked* for, not what the environment allowed:
    a batch that stops early on a death still says how far ahead the session was
    planning. A command may hold more than one call, and each ends at the shell's
    separator or at `--plan`, whose text is prose — counting inside it would make the
    batch size a function of the briefing.
    """
    total = 0
    for found in BATCH.finditer(command):
        for word in found.group(1).split():
            if word.startswith("-"):
                break
            if match := TOKEN.match(word):
                total += int(match.group(1)) if match.group(1) else 1
            else:
                break
    return total


# Every block the actuator wrote says which action of which batch it was, so the log
# knows the batch sizes whatever the session typed to get them.
STEP = re.compile(r"^action \d+ .*?\bstep 1/(\d+)\b", re.M)


def batches_from_log(ws: Path) -> list[int]:
    """How many actions each batch asked for, read off the log rather than the stream.

    The stream is the wrong source and this run showed why: the session wrote itself a
    `./go` wrapper on top of `./act do`, after which no command in the stream names a
    batch at all. The log is written by the actuator, so it says `step 1/12` whatever
    the session typed to get there.
    """
    sizes: list[int] = []
    for path in sorted(ws.glob("logs*.txt")):
        sizes += [int(n) for n in STEP.findall(path.read_text(errors="replace"))]
    return sizes


def session(stream: Path) -> dict:
    """Read one session's event stream.

    Two things about the stream that the arithmetic has to respect. One API response
    arrives as several `assistant` events — a thinking block, a text block, a tool
    call — all carrying the same `message.id` and the same usage, so summing events
    counts one request three times. And the `output_tokens` on those events is a
    snapshot taken before the response finished: over a whole session they add to 297
    where the result event reports 33,784. So the input side is read per message and
    the output side only in total, with each turn's *share* of it taken from the
    characters it actually produced.
    """
    got = {
        "turns": 0, "tool_calls": 0, "compactions": 0, "frames": 0,
        "cost": 0.0, "batches": [], "kinds": {}, "tools": {},
        "five_hour": 0.0, "seven_day": 0.0, "limit_status": "", "limits": 0,
        "turn_rows": [], "errors": 0, "result": "", "stop_reason": "",
        "duration_ms": 0, "num_turns": 0, "output_total": 0, "thinking_total": 0,
    }
    if not stream.exists():
        return got
    turns: dict[str, dict] = {}
    for line in stream.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "assistant":
            message = event.get("message", {})
            usage = message.get("usage", {})
            row = turns.setdefault(message.get("id", f"?{len(turns)}"), {
                "turn": len(turns) + 1,
                "calls": 0,
                "chars": 0,
                "input": usage.get("input_tokens", 0),
                "cache_read": usage.get("cache_read_input_tokens", 0),
                "cache_write": usage.get("cache_creation_input_tokens", 0),
                "at": event.get("timestamp", ""),
            })
            for block in message.get("content", []):
                shape = block.get("type")
                if shape in ("text", "thinking"):
                    # Thinking arrives as a signature and no text — 422KB of signature
                    # and zero characters of thought over the pilot's 119 blocks — so
                    # what a turn wrote can only be measured from its visible text and
                    # the commands it issued. In the one session that reported a total,
                    # thinking was 69% of the output tokens, so this proxy says which
                    # turns wrote more, not how much any of them wrote.
                    row["chars"] += len(block.get(shape) or "")
                    continue
                if shape != "tool_use":
                    continue
                row["chars"] += len(json.dumps(block.get("input", {})))
                row["calls"] += 1
                got["tool_calls"] += 1
                name = block.get("name", "?")
                got["tools"][name] = got["tools"].get(name, 0) + 1
                command = str(block.get("input", {}).get("command", ""))
                for label, pattern in KIND:
                    if pattern.search(command):
                        got["kinds"][label] = got["kinds"].get(label, 0) + 1
                        break
                else:
                    got["kinds"]["shell" if command else "file"] = (
                        got["kinds"].get("shell" if command else "file", 0) + 1
                    )
                if size := batch_size(command):
                    got["batches"].append(size)
        elif kind == "user":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") != "tool_result":
                    continue
                body = block.get("content")
                if isinstance(body, list):
                    got["frames"] += sum(1 for b in body if b.get("type") == "image")
                if block.get("is_error"):
                    got["errors"] += 1
        elif kind == "system" and event.get("subtype") == "compact_boundary":
            got["compactions"] += 1
        elif kind == "rate_limit_event":
            info = event.get("rate_limit_info", {})
            windows = info.get("unifiedWindows", {})
            got["five_hour"] = windows.get("five_hour", {}).get("utilization", 0.0)
            got["seven_day"] = windows.get("seven_day", {}).get("utilization", 0.0)
            got["limit_status"] = info.get("status", "")
            got["limits"] += 1
        elif kind == "result":
            got["cost"] = event.get("total_cost_usd", 0.0)
            got["result"] = str(event.get("result", ""))[:400]
            got["stop_reason"] = str(event.get("stop_reason") or event.get("subtype") or "")
            got["duration_ms"] = event.get("duration_ms", 0)
            got["num_turns"] = event.get("num_turns", 0)
            usage = event.get("usage", {})
            got["output_total"] = usage.get("output_tokens", 0)
            for model in (event.get("modelUsage") or {}).values():
                got["thinking_total"] += model.get("thinkingTokens", 0)
    got["turn_rows"] = list(turns.values())
    got["turns"] = len(turns)
    # The output side, distributed over the turns by what each of them wrote. An
    # apportionment, like the cost column it feeds, and marked as one wherever it is
    # printed: the total is measured, the split is not.
    # By what each turn visibly wrote — its text and its commands. Thinking is not in
    # the stream at all, so a turn that thought hard and typed little is undercounted.
    chars = sum(r["chars"] for r in got["turn_rows"]) or 1
    for row in got["turn_rows"]:
        row["output"] = got["output_total"] * row["chars"] / chars
    return got


def weighted(row: dict) -> float:
    return (
        WEIGHTS["input"] * row["input"]
        + WEIGHTS["cache_write"] * row["cache_write"]
        + WEIGHTS["cache_read"] * row["cache_read"]
        + WEIGHTS["output"] * row["output"]
    )


def when(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def depth(got: dict, bands: int = 5) -> list[list]:
    """Per-turn cost as the context grows.

    Bands are equal slices of the *turns*, not of the actions: what changes with depth
    is the context the model is charged for, and a session that slows down as it goes
    buys fewer actions per turn in the later bands, which is the thing to see.
    """
    rows = got["turn_rows"]
    if not rows:
        return []
    total = sum(weighted(r) for r in rows) or 1.0
    out = [["turns", "calls", "cache read/turn", "out/turn*", "s/turn", "$*"]]
    size = max(1, len(rows) // bands)
    for start in range(0, len(rows), size):
        band = rows[start:start + size]
        if not band:
            continue
        first, last = when(band[0]["at"]), when(band[-1]["at"])
        seconds = ((last - first).total_seconds() / len(band)) if first and last else 0.0
        share = sum(weighted(r) for r in band) / total
        out.append([
            f"{band[0]['turn']}-{band[-1]['turn']}",
            sum(r["calls"] for r in band),
            f"{sum(r['cache_read'] for r in band) / len(band) / 1000:.0f}k",
            f"{sum(r['output'] for r in band) / len(band):.0f}",
            f"{seconds:.0f}",
            f"{share * got['cost']:.2f}" if got["cost"] else f"{share:.0%}",
        ])
    return out


# --------------------------------------------------------------------------- #
# Saying it
# --------------------------------------------------------------------------- #


def table(rows: list[list]) -> str:
    if not rows:
        return "  (nothing)\n"
    width = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for n, row in enumerate(rows):
        out.append("  " + "  ".join(str(v).rjust(width[i]) for i, v in enumerate(row)))
        if n == 0:
            out.append("  " + "  ".join("-" * w for w in width))
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(prog="readout", description=__doc__.splitlines()[0])
    parser.add_argument("launch", help="a launch directory under ~/craftax-runs")
    parser.add_argument("--label", help="one workspace (default: all of them)")
    parser.add_argument("--no-replay", action="store_true",
                        help="skip the world side, which costs a minute of JAX")
    parser.add_argument("--step", type=int, default=250, help="curve resolution")
    args = parser.parse_args()

    root = Path(args.launch).resolve()
    labels = json.loads((root / RIG / "labels.json").read_text())
    for label in sorted(labels):
        if args.label and label != args.label:
            continue
        record = json.loads((root / ".envs" / label / "result.json").read_text())
        print(f"\n{'=' * 78}\n{root.name} / {label} — {record['variant']} seed "
              f"{record['seed']}, obs {'+'.join(record['obs'])}, "
              f"budget {record['actions_used']}/{record['budget']}\n{'=' * 78}")

        best = max(record["episode_scores"], default=0)
        print(
            f"\nscore\n"
            f"  mean life   {record.get('mean_episode', 0):>4}/226  "
            f"{record.get('mean_episode_pct', 0):>5.1f}%  "
            f"over {record.get('episodes_completed', 0)} finished lives\n"
            f"  best life   {best:>4}/226  {record['best_episode_pct']:>5.1f}%\n"
            f"  union       {record['score']:>4}/226  {record['score_pct']:>5.1f}%\n"
            f"  lives {record['lives']}, deaths {record['deaths']}, "
            f"level {record['max_level']}, cells {record['unique_cells']}\n"
            f"  per life    {record['episode_scores']}\n"
            f"  achievements {len(record['achievements'])}: "
            f"{' '.join(record['achievements'])}"
        )

        if not args.no_replay:
            rows = replay(record)
            print(f"\n{check(rows, record)}")
            out = root / RIG / f"trace-{label}.csv"
            with out.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nthe curve (per {args.step} actions)  ->  {out}")
            print(table(curve(rows, record["actions_used"], args.step)))
            print("what fired, and when")
            print(table(unlocks(rows)))

        got = session(root / label / "agent_stream.jsonl")
        batches = sorted(batches_from_log(root / label))
        actions = record["actions_used"]
        print("the session")
        print(f"  turns {got['turns']}, tool calls {got['tool_calls']}, "
              f"compactions {got['compactions']}, tool errors {got['errors']}")
        print(f"  by kind      {got['kinds']}")
        print(f"  by tool      {got['tools']}")
        if batches:
            print(f"  batches      {len(batches)}, "
                  f"median {batches[len(batches) // 2]} actions, "
                  f"mean {sum(batches) / len(batches):.1f}, "
                  f"largest {batches[-1]}")
            asked = sum(got["batches"])
            if asked < sum(batches):
                # A session is free to write itself a wrapper, and this one did. Worth
                # printing, because it is the difference between the two sources.
                print(f"  through ./act {asked} of {sum(batches)} actions were asked "
                      f"for by a command naming ./act itself")
        if got["tool_calls"]:
            print(f"  actions per tool call  {actions / got['tool_calls']:.1f}")
        print(f"  frames in context      {got['frames']}")
        print(f"  wall clock             {got['duration_ms'] / 60000:.0f} min")
        per = f"  (${1000 * got['cost'] / actions:.2f} per 1000 actions)" if actions else ""
        print(f"  cost                   ${got['cost']:.2f}{per}")
        print(f"  five-hour window       {got['five_hour']:.0%}"
              f"  (seven-day {got['seven_day']:.0%}, "
              f"status {got['limit_status'] or '?'})")
        print(f"  output                 {got['output_total']:,} tokens "
              f"({got['thinking_total']:,} of it thinking)")
        print(f"  ended                  {got['stop_reason'] or '?'}")
        if got["result"]:
            print(f"  said                   {got['result'][:200]}")
        print("\ncost at depth  (* apportioned from the session total, not measured "
              "per turn)")
        print(table(depth(got)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
