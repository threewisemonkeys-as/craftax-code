"""Describe observations with the model in `model/`.

    ./perceive                   the latest observation
    ./perceive A.png B.png ...   these frames, oldest first

The description is whatever `perceive()` in `model/perceive.py` returns for them.
"""

import importlib.util
import json
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


frames = [Path(a) for a in sys.argv[1:]]
if not frames:
    try:
        history = int(json.loads((ws / "model" / "VERSION").read_text())
                      .get("perception_history", 1))
    except (OSError, ValueError, AttributeError):
        history = 1
    found = sorted((ws / "frames").glob("[0-9]*.png"), key=order)
    if not found:
        print("there is no observation yet")
        sys.exit(1)
    frames = found[-max(1, history):]
missing = [str(f) for f in frames if not f.exists()]
if missing:
    print(f"no such file: {', '.join(missing)}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("model_perceive", model)
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
    print(module.perceive([str(f.resolve()) for f in frames]))
except Exception as exc:  # the model is learned code: say what broke, do not hide it
    print(f"model/perceive.py failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
