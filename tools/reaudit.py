#!/usr/bin/env python3
"""Grade a finished launch again with the audit as it is now.

A report's audit is written once, when its session ends, by whatever `rig/audit.py`
said then. When the audit learns something — as it did when it began reading what a
flagged call got back, and not only that it was made — every report written before
that is a verdict on a question it was never asked. This asks it.

Only the `audit` field is rewritten, in place, in the report and in the launch's
summary. Nothing is rotated: `write_report` rotates because a replay must not
overwrite the session it replays, and this is the same session, graded again.

    .env-venv/bin/python tools/reaudit.py ~/agent-runs/20260924-142020 [...]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "rig")]

import run  # noqa: E402
from agents import AGENTS  # noqa: E402
from audit import audit_session  # noqa: E402


def reaudit(root: Path) -> list[tuple[str, dict, dict]]:
    """Regrade every report in one launch; returns (label, old audit, new audit)."""
    rig = root / run.RIG
    labels = rig / "labels.json"
    siblings = list(json.loads(labels.read_text())) if labels.exists() else []
    changed = []
    for path in sorted((rig / "reports").glob("*.json")):
        if "-attempt" in path.stem:
            continue
        report = json.loads(path.read_text())
        label = report["label"]
        agent = AGENTS[report.get("agent", "claude")]
        stream = Path(report["workspace"]) / "agent_stream.jsonl"
        audit = audit_session(stream, agent, own=label, siblings=siblings or [label],
                              root=str(root), repo=str(run.REPO)).model_dump()
        changed.append((label, report.get("audit") or {}, audit))
        report["audit"] = audit
        path.write_text(json.dumps(report, indent=2))
    summary = rig / "summary.json"
    if summary.exists():
        audits = {label: new for label, _, new in changed}
        rows = json.loads(summary.read_text())
        for row in rows:
            if row.get("label") in audits:
                row["audit"] = audits[row["label"]]
        summary.write_text(json.dumps(rows, indent=2))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("launches", nargs="+", type=Path)
    for root in parser.parse_args().launches:
        for label, old, new in reaudit(root.expanduser()):
            verdict = ("LEAKED " + ", ".join(new["leaks"]) if new["leaks"]
                       else "attempted " + ", ".join(new["findings"]) if new["findings"]
                       else "clean")
            was = ", ".join(old.get("findings") or {}) or "clean"
            print(f"{root.name}/{label}: {verdict}   (was: {was})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
