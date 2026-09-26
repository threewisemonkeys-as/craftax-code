"""Describe observations with the model in `model/`.

    ./perceive                   the latest observation
    ./perceive A.png B.png ...   these frames, oldest first

The description is whatever `perceive()` in `model/perceive.py` returns for them.

A model whose VERSION says it is recursive is called perceive(prev, observation),
one frame at a time, `prev` being its own description of the frame before (""
before the first). With no frames named it runs over the current life, from its
first frame (or its last `burn_in` + 1 when VERSION sets one); the descriptions it
has already made are kept in `model/.states.json`, which goes when `model/` is
replaced. Named frames are run over in the order given, starting from "". The module
is loaded afresh for each frame, so `prev` is all it remembers.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ws = Path.cwd()
model = ws / "model" / "perceive.py"
if not model.exists():
    print("there is no model yet: `model/` appears the first time one has been learned")
    sys.exit(0)


def order(path: Path) -> tuple[int, bool]:
    # The frame an action produced, then the frame a new life began in, if any.
    return int(path.name[:6]), "-restart" in path.name


try:
    version = json.loads((ws / "model" / "VERSION").read_text())
    if not isinstance(version, dict):
        version = {}
except (OSError, ValueError):
    version = {}
recursive = version.get("recursive")

frames = [Path(a) for a in sys.argv[1:]]
named = bool(frames)
if not frames:
    found = sorted((ws / "frames").glob("[0-9]*.png"), key=order)
    if not found:
        print("there is no observation yet")
        sys.exit(1)
    if recursive:
        starts = [i for i, f in enumerate(found) if "-restart" in f.name]
        frames = found[starts[-1] if starts else 0:]  # this life
        burn_in = recursive.get("burn_in") if isinstance(recursive, dict) else None
        if burn_in is not None:
            frames = frames[-(int(burn_in) + 1):]
    else:
        try:
            history = int(version.get("perception_history", 1))
        except (TypeError, ValueError):
            history = 1
        frames = found[-max(1, history):]
missing = [str(f) for f in frames if not f.exists()]
if missing:
    print(f"no such file: {', '.join(missing)}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_perceive", model)
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
    if not recursive:
        print(module.perceive([str(f.resolve()) for f in frames]))
        sys.exit(0)
    # Kept states are only reused for a whole life, scanned from its first frame, by
    # this very model: a burn-in or a named list starts the recursion somewhere else.
    cache_path = ws / "model" / ".states.json"
    reuse = not named and not (isinstance(recursive, dict)
                               and recursive.get("burn_in") is not None)
    stamp = hashlib.sha256(model.read_bytes()).hexdigest()
    kept = {}
    if reuse:
        try:
            saved = json.loads(cache_path.read_text())
            if saved.get("model") == stamp and saved.get("first") == frames[0].name:
                kept = saved.get("states", {})
        except (OSError, ValueError, AttributeError):
            kept = {}
    prev, states = "", {}
    for f in frames:
        if f.name in kept:
            prev = kept[f.name]
        else:
            # afresh per frame, as in training: `prev` is the model's only memory
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            prev = module.perceive(prev, str(f.resolve()))
            prev = prev if isinstance(prev, str) else str(prev)
        states[f.name] = prev
    if reuse and states != kept:
        tmp = cache_path.with_name(f".states.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"model": stamp, "first": frames[0].name,
                                   "states": states}))
        os.replace(tmp, cache_path)
    print(prev)
except SystemExit:
    raise
except Exception as exc:  # the model is learned code: say what broke, do not hide it
    print(f"model/perceive.py failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
