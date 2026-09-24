import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

# Every process that builds a game pays ~37s before its first step, and the suite
# starts a dozen of them: each daemon test spawns its own, and so does each xdist
# worker. About 17s of that is XLA compiling `reset`, `step` and the renderer, which
# is the same work every time, so it is cached on disk across processes and runs.
# Set here as environment variables rather than through `jax.config` so the daemons
# the tests spawn inherit it; set before anything imports jax.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(REPO / ".jax-cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
