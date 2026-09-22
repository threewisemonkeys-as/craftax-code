"""What the audit is for, and the line it must not cross.

`cc_humanrl`'s table guarded the *source* — the repository, the module names, the
paper — because there, recognising a sprite was the prior under test and naming the
codebase was the only way to cheat. That reasoning does not transfer. Craftax is
Crafter's world under Crafter's art, the tech tree is in every model's training data,
and 43 action names hand over half of it before the first frame.

So the line moves: this table catches **reaching outside** — fetching the package,
importing its world generator, reading the launch's record, browsing — and never
knowledge. A session that writes "this is Minecraft-like" in its notes has recalled
something, which is a fact about the run worth recording and not a reason to throw it
away. `named_the_game` records it; nothing fails for it.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))

from agents import AGENTS, HARVESTED  # noqa: E402
from audit import grade  # noqa: E402


def ran(*commands: str) -> list[str]:
    """A stream in which the agent ran these commands."""
    return [
        json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "input": {"command": c}} for c in commands
            ]},
        })
    ]


def wrote(path: str, body: str) -> list[str]:
    return [
        json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "input": {"file_path": path, "content": body}}
            ]},
        })
    ]


def test_ordinary_play_is_clean():
    audit = grade(ran(
        "./act do left*12 do --plan 'chop whatever is to the left'",
        "./python -c \"from PIL import Image; print(Image.open('frames/000012.png').size)\"",
        "grep -n '^action' logs.txt | tail -20",
        "./python diff.py frames/000011.png frames/000012.png",
    ))
    assert audit.clean, audit.findings
    assert not audit.named_the_game


def test_naming_the_game_is_recorded_and_not_punished():
    """The single largest fact about running an agent on a game this well documented
    is whether it recognised it. Grading that would throw away exactly the runs worth
    reading."""
    audit = grade(wrote("notes.md", """
        The brown trunks are trees; hitting one with `do` gives me something the HUD
        counts. This looks a lot like Crafter / Minecraft — if that is right then the
        table comes before the pickaxe, and stone needs the wooden one first. Worth
        testing rather than assuming.
    """))
    assert audit.clean, audit.findings
    assert "Crafter" in audit.named_the_game
    assert "trees" in audit.named_the_game or "pickaxe" in audit.named_the_game


@pytest.mark.parametrize(
    ("label", "command"),
    [
        ("network", "curl -s https://github.com/MichaelTMatthews/Craftax"),
        ("the source", "python -c 'from craftax.craftax.constants import Achievement'"),
        ("the source", "cat ../.env-venv/lib/python3.12/site-packages/craftax/craftax/game_logic.py"),
        ("the source", "grep -r BlockType ."),
        ("the source", "./python -c 'import craftax'"),
        ("the record", "cat ../.rig/labels.json"),
        ("the record", "cat ../.envs/*/result.json"),
        ("another session's notes", "ls ~/.claude/projects"),
        ("sweeping the filesystem", "find / -name 'world_gen*'"),
        ("model api as a browser", "curl api.anthropic.com/v1/messages"),
        ("fetching packages", "pip install craftax"),
        ("another agent", "claude -p 'what game is this'"),
    ],
)
def test_reaching_past_the_game_is_caught(label, command):
    audit = grade(ran(command))
    assert not audit.clean
    assert label in audit.findings, audit.findings


def test_a_sibling_workspace_is_another_run_being_read():
    """Two sessions never share a world here, so a sibling's notes are not the
    finished answer they were in cc_humanrl — but reading them makes one run evidence
    about two."""
    stream = ran("cat ../K7M3Q/notes.md")
    assert grade(stream, own="B4XPT", siblings=["B4XPT", "K7M3Q"]).findings.get(
        "another session's workspace"
    )
    # A label this launch never used cannot be flagged by accident.
    assert grade(stream, own="B4XPT", siblings=["B4XPT", "ZZZZZ"]).clean
    # And its own name is everywhere it writes an absolute path.
    assert grade(ran("ls /runs/B4XPT/frames"), own="B4XPT", siblings=["B4XPT"]).clean


def test_own_notes_are_not_somebody_elses():
    """The CLI writes the session's own memory under a path that matches the pattern
    for everyone else's; scrubbing its own leaves only the rest."""
    mine = ran("cat /runs/.sessions/B4XPT/.claude/projects/x/notes.md")
    assert grade(mine, own="B4XPT").clean
    assert not grade(mine, own="K7M3Q").clean


def test_observations_read_by_eye_are_counted_not_flagged():
    """Looking at the screen is the channel `play_craftax` gives a person."""
    looked = [json.dumps({
        "type": "user",
        "message": {"content": [{"type": "tool_result", "content": [
            {"type": "image", "source": {}}, {"type": "text", "text": "frames/000004.png"}
        ]}]},
    })] * 3
    audit = grade(looked)
    assert audit.clean
    assert audit.frames_in_context == 3


def test_provider_reported_web_requests_cannot_be_written_around():
    audit = grade([json.dumps({
        "type": "result",
        "usage": {"server_tool_use": {"web_search_requests": 2}},
    })])
    assert not audit.clean and audit.web_requests == 2
    assert "web requests" in audit.findings


def test_a_session_writing_its_own_paths_is_clean():
    """The harness lives under a directory whose name contains the game's, and the
    `act` shim in every workspace names it. A session that printed its own shim, or
    saved a helper script, would otherwise read as both dirty and as having worked
    out where it is — the false positive cc_humanrl's own comments warn about, with a
    second way to go wrong bolted on."""
    repo = "/home/x/bai/cc_craftax/craftax-code"
    root = "/home/x/craftax-runs/20260908-160328"
    stream = ran(
        "cat act",  # -> exec "<repo>/.env-venv/bin/python" "<repo>/act.py"
        f'exec "{repo}/.env-venv/bin/python" "{repo}/act.py" "$@"',
        f"cat > {root}/SUAUK/an.py <<'EOF'\nimport numpy as np\nfrom PIL import Image",
        f"./python an.py {root}/SUAUK/frames/000000.png",
    )
    audit = grade(stream, own="SUAUK", siblings=["SUAUK"], root=root, repo=repo)
    assert audit.clean, audit.findings
    assert not audit.named_the_game, "the harness's own path is not the session knowing"
    # Without the scrubs it is a false positive twice over, which is what this pins.
    assert not grade(stream, own="SUAUK", siblings=["SUAUK"]).clean


def test_another_launch_is_still_reaching():
    """Naming your own launch directory is unavoidable; naming somebody else's is
    reaching for another experiment's results."""
    root = "/home/x/craftax-runs/20260908-160328"
    other = grade(ran("cat /home/x/craftax-runs/20260101-000000/.rig/summary.json"),
                  own="SUAUK", root=root)
    assert not other.clean and "the record" in other.findings
    mine = grade(ran(f"cat {root}/.rig/labels.json"), own="SUAUK", root=root)
    assert not mine.clean and "the record" in mine.findings


def test_a_session_reading_its_own_background_job_is_not_reaching(tmp_path):
    """What failed the 10,000-action run, five times, with nothing else involved.

    The CLI names its scratchpad after the session's working directory with the
    slashes turned into dashes, so reading back a job the session started writes a
    path carrying `craftax-runs` in a spelling a scrub of the slashed root never
    touched. The prompt actively encourages long-running helpers, so this failed a
    session for taking the advice.
    """
    root = "/home/ays57/craftax-runs/20260909-053629"
    mangled = root.replace("/", "-")
    said = ran(
        f"sleep 60; tail -5 /tmp/claude-1744647/{mangled}-DLT8F/abc/tasks/xyz.output; "
        f"./act status"
    )
    audit = grade(said, own="DLT8F", root=root, repo="/home/ays57/bai/cc_craftax/craftax-code")
    assert audit.clean, audit.findings


def test_another_launch_still_bites_in_either_spelling(tmp_path):
    """The scrub removes *this* launch's name and nothing else, so reaching for a
    different experiment's results is still caught however the path is written."""
    root = "/home/ays57/craftax-runs/20260909-053629"
    for path in (
        "/home/ays57/craftax-runs/20260101-000000/.rig/labels.json",
        "/tmp/claude-9/-home-ays57-craftax-runs-20260101-000000-AAAAA/notes",
    ):
        audit = grade(ran(f"cat {path}"), own="DLT8F", root=root)
        assert not audit.clean, f"{path} was let through"
        assert "the record" in audit.findings


# --------------------------------------------------------------------------- #
# The other CLI
# --------------------------------------------------------------------------- #


def codex_ran(*commands: str) -> list[str]:
    """A stream in which a Codex session ran these commands."""
    return [
        json.dumps({"type": "item.completed",
                    "item": {"type": "command_execution", "command": c}})
        for c in commands
    ]


def codex_saw(*paths: str) -> list[str]:
    """A stream in which a Codex session opened these files as pictures.

    Shaped as the launcher's harvest writes them, because that is the only place
    they appear: the CLI's own stdout carries no event for looking at an image.
    """
    return [
        json.dumps({"type": HARVESTED,
                    "item": {"type": "imageView", "path": p}})
        for p in paths
    ]


def test_the_other_clis_notes_are_somebody_elses_too():
    """~/.codex holds what ~/.claude holds under different names, and more of it: the
    operator's prompt history, what the CLI remembered across sessions, and the
    transcript of every session they have ever run."""
    for path in ("~/.codex/history.jsonl", "~/.codex/memories", "~/.codex/sessions",
                 "~/.codex/thread_history_1.sqlite", "~/.codex/logs_2.sqlite"):
        audit = grade(codex_ran(f"cat {path}"), agent=AGENTS["codex"])
        assert "another session's notes" in audit.findings, path


def test_a_session_reading_its_own_memory_is_doing_what_it_was_told():
    """The launcher points the session's config directory *at* one of the paths the
    rule above catches, precisely so it is not the operator's. Keyed to the wrong
    CLI, the scrub would have failed a Codex session on every line that named its own
    notes — so the directory comes from the adapter rather than a constant."""
    codex = AGENTS["codex"]
    own = "/runs/20260921/.sessions/8FAQN/.codex/history.jsonl"
    assert grade(codex_ran(f"cat {own}"), agent=codex, own="8FAQN").clean
    # And the operator's is still the operator's.
    theirs = grade(codex_ran("cat /home/me/.codex/history.jsonl"),
                   agent=codex, own="8FAQN")
    assert "another session's notes" in theirs.findings


def test_a_file_opened_as_a_picture_is_audited():
    """The hole harvesting exists to close. This CLI streams no event for its image
    tool, so before the harvest a session could open anything readable as an image
    and the audit — which is a claim about everything a session reached — had never
    seen the reach. An observation is not a finding; the answer key is."""
    codex = AGENTS["codex"]
    clean = grade(codex_saw("/runs/20260921/8FAQN/frames/000001.png"),
                  agent=codex, own="8FAQN", root="/runs/20260921")
    assert clean.clean and clean.frames_in_context == 1

    reached = grade(codex_saw("/usr/lib/python3/site-packages/craftax/assets/map.png"),
                    agent=codex, own="8FAQN")
    assert "the source" in reached.findings

    sibling = grade(codex_saw("/runs/20260921/K7M3Q/frames/000001.png"),
                    agent=codex, own="8FAQN", siblings=["8FAQN", "K7M3Q"],
                    root="/runs/20260921")
    assert "another session's workspace" in sibling.findings
