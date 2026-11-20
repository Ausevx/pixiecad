# Contributing to PixieCAD

Pull requests are welcome. Here is what you need to know to get one merged.

## Development setup

```bash
git clone https://github.com/Ausevx/pixiecad.git
cd pixiecad
python3 -m venv .venv
.venv/bin/pip install -e ".[mesh,sfm,web,dev]"
```

## Running the tests

```bash
.venv/bin/python -m pytest -q
```

The default run is 599 tests in about 90 seconds, fully offline — no GPU, no
network, no API keys. See the [Tests section in the README](README.md#tests)
for the slow/network marker tiers.

## Code style

- Python: standard library style, 4-space indent, `snake_case`.
- TypeScript: the existing config in `frontend/tsconfig.json`.
- Keep new dependencies minimal and optional where possible — the base install
  should stay importable without trimesh or pycolmap.

## Commit messages

One line summary in the imperative mood, present tense:

```
Fix sparse reconstruction crash on <16 input photos
Add --drawings flag to dashboard
```

## What makes a good PR

- **Narrow scope**: one concern per PR. It is much easier to review and
  merge a small, well-scoped change than a large one.
- **Tests**: new behaviour should have tests. Bug fixes should have a
  regression test that fails before the fix and passes after.
- **No new hard deps**: if your feature needs a new package, put it in
  an optional extra in `pyproject.toml` with a sensible import guard.
