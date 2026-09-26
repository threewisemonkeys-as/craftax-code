"""What the world-model arm learned at each pause, for the replay page.

A run launched with `--pause-hook` stops at its pauses and hands the run so far to a
learner; the harness knows nothing about how. What it keeps is the learner's answer —
`.rig/updates/<label>.jsonl`, one line per update, and the JSON result and log the hook
wrote beside it (`.rig/updates/<label>/NNN.*`). That much is the harness's own and is
read for any hook.

The learner that ships with bai (`online_learning/update.py`) also leaves its working
behind, in `<env_dir>/online/updates/NNN/`: the search it ran, every candidate model,
what each reflection was told and what it answered. When that is there the page shows
the search step by step. When it is not — another hook, or an update still running —
the page shows what the record says and nothing it would have to guess.

A run without a pause hook has none of this, and `learning()` returns [] for it: the
page then has no Learning section at all.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

# A candidate model is two texts, the perception module and the world knowledge, and
# every update writes several of them. Bounded, because four updates of four steps
# each is already a megabyte of the page, and unbounded, one 60k-character module
# edited four times would be most of it.
CAP_TEXT = 60_000
CAP_DIFF = 24_000
CAP_FIELD = 2_500
CAP_PROSE = 12_000
CAP_LOG = 8_000
COMPONENTS = ("perception", "world_knowledge")

IT_LINE = re.compile(
    r"it=(\d+): parent (\d+) \(h=([\d.]+)\) -> child (\d+) h=([\d.]+) \[([^\]]*)\]")
SEED_LINE = re.compile(r"seed full-train score h0=([\d.]+)")
FENCE = re.compile(r"```[^\n]*\n.*?```", re.S)


def _jsonl(path: Path) -> list[dict]:
    """Every line that parses. A file being appended to can end in half a line."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            got = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(got, dict):
            out.append(got)
    return out


def _json(path: Path) -> dict:
    try:
        got = json.loads(path.read_text())
        return got if isinstance(got, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _tail(path: Path, cap: int = CAP_LOG) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return text if len(text) <= cap else "…\n" + text[-cap:]


def _cap(text, cap: int) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= cap else text[:cap] + f"\n… [{len(text) - cap} more characters]"


def diff(before: str, after: str) -> dict:
    """A unified diff, and how many lines it adds and drops."""
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                      "before", "after", n=2, lineterm=""))[2:]
    added = sum(1 for line in lines if line.startswith("+"))
    dropped = sum(1 for line in lines if line.startswith("-"))
    return {"text": _cap("\n".join(lines), CAP_DIFF), "added": added, "dropped": dropped}


def _lines(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if len(line.strip()) > 20}


def match_calls(steps: list[dict], calls: list[dict]) -> dict[int, dict]:
    """Which reflection call wrote which step's proposal.

    Not by order, and not by the call's `component`: reflections run several at a
    time and are logged as they finish, and the label a call carries has been seen
    naming the other component. The proposal is a verbatim part of the answer that
    produced it, so each step goes to the call whose answer holds most of its lines.
    """
    scored = []
    for a, step in enumerate(steps):
        wanted = _lines("\n".join(str(v) for v in (step.get("proposed") or {}).values()))
        if not wanted:
            continue
        for b, call in enumerate(calls):
            share = len(wanted & _lines(str(call.get("response", "")))) / len(wanted)
            if share > 0.5:
                scored.append((share, a, b))
    out, used = {}, set()
    for _, a, b in sorted(scored, reverse=True):
        if a in out or b in used:
            continue
        out[a] = calls[b]
        used.add(b)
    return out


def prose(response: str, component: str) -> str:
    """The reflector's account of its change, without the change itself.

    The new module is in the answer as a fenced block tens of thousands of characters
    long, and the page shows it as a diff against its parent instead.
    """
    kept = FENCE.sub(lambda m: f"[the new {component}: {m.group(0).count(chr(10))} lines, "
                               f"shown as a diff]", response)
    # A world-knowledge answer is often the whole new document with a line of preamble;
    # its prose is then most of it, and the diff says the rest better.
    return _cap(kept.strip(), CAP_PROSE)


def evidence(record: dict, frame_of: dict[str, str]) -> dict:
    """One example a reflection was shown, as fields the page can lay out.

    The learner's record is a nested dict whose keys are its own business, so this
    keeps them as they were written. A value that is a path to one of the run's
    frames becomes that frame, which the page can put a picture to.
    """
    fields, frames = [], []
    for section, value in record.items():
        items = value.items() if isinstance(value, dict) else [(section, value)]
        for key, text in items:
            text = "" if text is None else str(text)
            name = frame_of.get(text.strip())
            if name:
                frames.append([re.sub(r"\s*\(.*$", "", key), name])
                continue
            fields.append([key if isinstance(value, dict) else section,
                           _cap(text, CAP_FIELD)])
    return {"fields": fields, "frames": frames}


def search(out: Path, frame_of: dict[str, str]) -> dict | None:
    """The learner's search, read from what it left in `online/updates/NNN/`.

    The tree comes from the learner's log, which is written as each node is scored and
    so is there while an update is still running; the candidates, what each step was
    told and what it answered are written when the search ends.
    """
    if not out.is_dir():
        return None
    log = out / "rexpure.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    runs = sorted(out.glob("rexpure_run_seed*"))
    run_dir = runs[-1] if runs else out
    cands = {int(c["idx"]): c for c in _jsonl(run_dir / "candidates.jsonl") if "idx" in c}
    steps = _jsonl(run_dir / "process_log.jsonl")
    calls = _jsonl(run_dir / "reflection_calls.jsonl")

    seed = SEED_LINE.search(text)
    nodes = [{"idx": 0, "parent": None, "score": float(seed[1]) if seed else
              (cands[0]["train_score"] if 0 in cands else None), "edited": [], "step": 0}]
    for m in IT_LINE.finditer(text):
        nodes.append({"idx": int(m[4]), "parent": int(m[2]), "score": float(m[5]),
                      "edited": [c.strip() for c in m[6].split(",") if c.strip()],
                      "step": int(m[1])})
    if len(nodes) == 1 and len(cands) > 1:  # a learner that logs differently
        nodes = [{"idx": i, "parent": (c.get("parents") or [None])[0],
                  "score": c.get("train_score"), "edited": [], "step": i}
                 for i, c in sorted(cands.items())]
    for node in nodes:
        cand = cands.get(node["idx"])
        node["chars"] = ({c: len(str(cand.get(c) or "")) for c in COMPONENTS}
                         if cand else {})

    by_call = match_calls(steps, calls)
    shown = []
    for a, step in enumerate(steps):
        edited = list(step.get("components") or [])
        child, parent = cands.get(step.get("new_idx")), cands.get(step.get("selected"))
        changes = {}
        for comp in edited:
            after = (step.get("proposed") or {}).get(comp)
            if after is None and child:
                after = child.get(comp)
            if parent is not None and after is not None:
                changes[comp] = diff(str(parent.get(comp) or ""), str(after))
        call = by_call.get(a)
        told = []
        for comp, records in (step.get("feedback") or {}).items():
            for record in records or []:
                if isinstance(record, dict):
                    told.append({"component": comp, **evidence(record, frame_of)})
        shown.append({
            "i": step.get("i", a + 1), "parent": step.get("selected"),
            "parent_score": step.get("selected_score"), "child": step.get("new_idx"),
            "score": step.get("new_score"), "edited": edited,
            "verdict": step.get("verdict"), "accepted": step.get("accepted"),
            "rows": len(step.get("minibatch_ids") or []),
            "told": told, "changes": changes,
            "prompt_chars": len(str(call.get("prompt", ""))) if call else 0,
            "account": prose(str(call.get("response", "")), ", ".join(edited))
            if call else "",
        })

    header = {}
    for key, pattern in (("task_lm", r"task_lm \(F\) = (\S+)"),
                         ("reflection_lm", r"reflection_lm = (\S+)"),
                         ("rows", r"train=(\d+)")):
        if found := re.search(pattern, text):
            header[key] = found[1]
    return {"nodes": nodes, "steps": shown, "log": _tail(log), "done": bool(cands),
            **header}


def shipped(out: Path, idx, had: dict | None) -> dict:
    """The model this update shipped, and what changed from the one the agent had.

    Read from the candidate the result names, which is the model *before* the hook
    redacted the game's name out of it — the result says how many redactions there
    were. The candidate the search started from is the model the agent had.
    """
    runs = sorted(out.glob("rexpure_run_seed*"))
    cands = {int(c["idx"]): c for c in _jsonl(runs[-1] / "candidates.jsonl")} if runs else {}
    now = cands.get(idx) if idx is not None else None
    if not now:
        return {}
    was = had or cands.get(0) or {}
    return {
        comp: {"text": _cap(now.get(comp) or "", CAP_TEXT),
               "chars": len(str(now.get(comp) or "")),
               "diff": diff(str(was.get(comp) or ""), str(now.get(comp) or ""))}
        for comp in COMPONENTS
    }


def learning(root: Path, rig: str, label: str, record: dict) -> list[dict]:
    """Every update this run has had, and the one it is having, oldest first.

    [] for a run that never paused to learn, which is how the page knows to leave the
    section out.
    """
    here = root / rig / "updates"
    lines = _jsonl(here / f"{label}.jsonl")
    pauses = list(record.get("pauses") or [])
    if not lines and not (pauses and (here / label).is_dir()):
        return []
    env = Path(record.get("env_dir") or root / ".envs" / label)
    online = env / "online"
    frame_of = {v: k for k, v in reversed(list(_json(online / "frame_index.json").items()))}

    out = []
    done = {int(line.get("update", 0)) for line in lines}
    # An update still running has a pause in the record and no line yet: shown as
    # such, with whatever the learner has written so far.
    running = [{"update": k, "at": p.get("at"), "events": p.get("events"), "running": True}
               for k, p in enumerate(pauses, start=1) if k not in done]
    for line in lines + running:
        k = int(line.get("update", 0))
        result = _json(here / label / f"{k:03d}.json")
        work = online / "updates" / f"{k:03d}"
        ship = result.get("ship") or {}
        out.append({
            "k": k, "at": line.get("at"), "running": bool(line.get("running")),
            "exit": line.get("exit"), "shipped": line.get("shipped"),
            "cost": line.get("cost_usd"), "seconds": line.get("seconds"),
            "events": line.get("events") or [],
            "buffer": result.get("buffer"), "composition": result.get("composition"),
            "prequential": result.get("prequential"), "costs": result.get("costs"),
            "best": result.get("best_train_score"), "ship": ship,
            "screened": result.get("screened") or [], "error": result.get("error"),
            "hook_log": _tail(here / label / f"{k:03d}.log"),
            "search": search(work, frame_of),
            "model": shipped(work, ship.get("idx"), None) if line.get("shipped") else {},
        })
    return sorted(out, key=lambda u: u["k"])
