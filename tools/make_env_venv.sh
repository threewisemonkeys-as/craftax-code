#!/bin/sh
# The interpreter the harness runs under.
#
# Craftax is JAX, and JAX is not something to add to the parent project's
# dependency set for one experiment's sake -- it would sit in the same venv as
# the rest of bai's ML stack and resolve against it. So the harness gets its own,
# here, and run.py is launched with it: act.py inherits it through the workspace
# shim, which is how the daemon ends up holding a JAX process.
#
# The version is pinned, and the pin is checked rather than trusted. Every number
# in notes/craftax-harness-plan.md was measured against 1.6.1; 1.6.0 changed how
# Craftax-Classic's reward handles lava damage and the end-of-episode health
# penalty, so a silent upgrade would move the denominator underneath the results.
set -e
cd "$(dirname "$0")/.."
uv venv --python 3.12 .env-venv
uv pip install --python .env-venv/bin/python \
    "craftax==1.6.1" "pydantic>=2" python-dotenv pytest
# Building the texture cache is a one-minute job that otherwise happens inside
# the first session, on the clock, in a directory the agent is watching.
.env-venv/bin/python - <<'WARM'
import jax
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Action, Achievement
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN as HUMAN
from craftax.craftax.renderer import make_craftax_pixel_renderer as full_renderer
from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN as CLASSIC_HUMAN
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer as classic

for name, renderer, size in (
    ("Craftax-Symbolic-v1", full_renderer, HUMAN),
    ("Craftax-Classic-Symbolic-v1", classic, CLASSIC_HUMAN),
):
    env = make_craftax_env_from_name(name, auto_reset=False)
    _, state = env.reset(jax.random.PRNGKey(0), env.default_params)
    print(f"{name}: warm, frame {renderer(size)(state).shape}")
print(f"{len(Action)} actions, {len(Achievement)} achievements")
WARM
echo "harness interpreter ready: $(pwd)/.env-venv/bin/python"
