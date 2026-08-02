#!/usr/bin/env python3
"""Print the available Claude Code skills, for a SessionStart hook.

Why this exists: the skills listing is injected into the agent's context by
the harness as scaffolding, not as conversation. When a long session is
compacted, the summary carries the conversation forward but not the
scaffolding -- so the agent comes back knowing the project but no longer
knowing which skills exist, and (correctly) refuses to guess skill names.

That is exactly what happened in this project: `agy-orchestrate` was
installed and well-formed the whole time, but after a compaction the agent
could not see it and said so rather than inventing a name.

`bd prime` already solves the same problem for the issue tracker by re-running
on SessionStart. This does the equivalent for skills, so a compacted session
recovers both. Reads only the frontmatter, so it stays fast enough for a hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Long descriptions (some run to several hundred words of trigger phrases)
# would swamp the context this is meant to cheaply restore.
MAX_DESCRIPTION = 240


def _frontmatter(path: Path) -> dict[str, str]:
    """Parse the leading YAML block. Deliberately minimal: no yaml dependency.

    Only top-level `key: value` pairs matter here, and a hook that crashes on
    an odd file is worse than one that skips it.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}

    out: dict[str, str] = {}
    key = None
    for line in text.split("\n")[1:]:
        if line.strip() == "---":
            break
        if line[:1] in (" ", "\t") and key:
            # Continuation of a folded value.
            out[key] += " " + line.strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            out[key] = value.strip()
    return out


def _skill_dirs() -> list[Path]:
    roots = [Path.home() / ".claude" / "skills", Path.cwd() / ".claude" / "skills"]
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            # resolve() so symlinked skills (a shared .agents/skills tree) are
            # followed rather than skipped.
            if (entry / "SKILL.md").is_file() or (entry.resolve() / "SKILL.md").is_file():
                found.append(entry)
    return found


def main() -> int:
    dirs = _skill_dirs()
    if not dirs:
        print("No skills installed.")
        return 0

    print(f"# Available skills ({len(dirs)})")
    print()
    print("Invoke with the Skill tool, or the user may type /<name>.")
    print()
    for entry in dirs:
        meta = _frontmatter(entry / "SKILL.md")
        if not (entry / "SKILL.md").is_file():
            meta = _frontmatter(entry.resolve() / "SKILL.md")
        name = meta.get("name") or entry.name
        description = meta.get("description", "").strip()
        if len(description) > MAX_DESCRIPTION:
            description = description[:MAX_DESCRIPTION].rsplit(" ", 1)[0] + " ..."
        print(f"- **{name}** — {description}" if description else f"- **{name}**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
