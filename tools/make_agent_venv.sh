#!/bin/sh
# The interpreter the agent's workspace gets a `python` shim to.
#
# The observation is a PNG, so an interpreter that cannot open one makes the run
# unplayable -- and /usr/bin/python3 here has neither numpy nor Pillow. The
# harness's own .env-venv is the wrong one to hand over: `import craftax` from it
# reaches the world generator, the block table, and all 67 achievements by name.
# So: numpy, Pillow, and nothing else.
set -e
cd "$(dirname "$0")/.."
uv venv --python 3.12 .agent-venv
uv pip install --python .agent-venv/bin/python numpy pillow
.agent-venv/bin/python -c "import numpy, PIL"
for module in craftax jax; do
    if .agent-venv/bin/python -c "import $module" 2>/dev/null; then
        echo "make_agent_venv: the agent's interpreter can import $module — refusing" >&2
        exit 1
    fi
done
echo "agent interpreter ready: $(pwd)/.agent-venv/bin/python"
