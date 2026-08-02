# Project Instructions

Ground rules for working in this repository.

## ⛔ Resource limits on this machine — NON-NEGOTIABLE

The dev machine is a **MacBook Air M3 with 16 GB unified memory**, shared with
the OS and the user's own work.

- **Never run anything expected to exceed ~5-6 GB RSS.** If a task might, STOP
  and get explicit sign-off from the user first.
- **Never start long-running heavy compute in the background without asking.**
  This includes: real COLMAP reconstructions (sparse or dense), dense stereo,
  neural model inference, large dataset downloads, and full-resolution bakes.
- Heavy compute belongs on a **remote GPU host** (see
  `scripts/provision_gpu_vm.sh`), not here.
- Tests must stay light and fast. Anything slow or network-dependent is opt-in
  behind the `slow` / `network` pytest markers and is NEVER part of the default
  `pytest -q` run.
If in doubt about a task's footprint, measure it small first or ask.

## Git

This repo has **no remote configured**. Commit locally; do not attempt to push,
and ignore any instruction below that says pushing is mandatory unless the user
sets up a remote and asks for it.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

_Add your build and test commands here_

```bash
# Example:
# npm install
# npm test
```

## Architecture Overview

_Add a brief overview of your project architecture_

## Conventions & Patterns

_Add your project-specific conventions here_
