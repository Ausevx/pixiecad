# PixieCAD

Turn photos of an object into a 3D model with a polygon budget you control.
Works two ways: **measure** a real object from many photos (photogrammetry),
or **generate** a plausible model from a few photos, real or AI-made.

## Project structure

```
pixiecad/
├── CLAUDE.md              dev-machine constraints: 16GB memory limit, no git remote
├── PLAN.md                architecture notes / design log
├── pyproject.toml         deps, package config
├── scripts/
│   └── provision_gpu_vm.sh   stand up a GCP GPU VM for dense/generative stages
├── images/                sample F1 test set (generated, for pipeline testing)
├── src/pixiecad/
│   ├── cli.py             all commands (see below)
│   ├── spec.py            ObjectSpec — name, target_faces, dimensions, scale source
│   ├── workspace.py        content-addressed stage cache (every stage is resumable)
│   ├── pipeline.py         run_build — chains every stage below into one call
│   │
│   ├── ingest/             S0  — blur/exposure score, mirror-aware dedup, downscale
│   ├── scale/               S1  — ArUco marker or declared-dimension → metric scale
│   ├── geometry/
│   │   ├── sparse.py        S2a — pycolmap: camera poses + sparse point cloud
│   │   └── dense.py         S2b — COLMAP dense stereo, runs in Docker on a GPU host
│   ├── generative/          S3  — image→mesh when there aren't enough photos to
│   │   ├── base.py            triangulate (fal.ai backend, self-hosted TRELLIS
│   │   ├── fal_backend.py     backend, or the "fake" backend used in tests)
│   │   └── trellis_remote.py
│   ├── meshops/
│   │   ├── cleanup.py       S4  — weld verts, drop degenerate faces, remove floaters
│   │   ├── decimate.py      S6a — exact-face-count decimation (pyfqmr)
│   │   ├── bake.py          S6b — UV unwrap (xatlas) + object-space normal bake
│   │   └── ortho.py         S7  — orthographic front/side/top SVG line drawings
│   ├── parts/                S5  — split one mesh into separate components
│   │   ├── segment.py         (connected-components or k-means clustering)
│   │   └── export.py          (per-part budget, per-part .glb, manifest.json)
│   ├── vision/              S0.5 — optional LLM layer: photo triage + part naming.
│   │   ├── base.py            Works with NO api key (geometric fallback names,
│   │   └── triage.py          structural coverage checks); a key just adds words.
│   │
│   ├── executors/           local or SSH+GPU-host job execution (rsync + docker)
│   ├── cloud.py             gcloud status, running instances, spend estimate
│   ├── export.py            final .glb writer (mesh + UV + normal map + provenance)
│   └── web/                 FastAPI dashboard + static/index.html (three.js viewer)
│
└── tests/                  154 tests, ~15s, no network/GPU required
```

### The two paths through the pipeline

| Input | Stages run | Output |
|---|---|---|
| **16+ photos**, real object, full orbit | S0→S1→S2a→S2b→S4→S6→S7 | Metrically accurate — the real dimensions you declared |
| **1–15 photos**, real or AI-generated | S0→S3(generative)→S4→S6→S7 | Plausible shape; unseen surfaces are invented, flagged in the output metadata |

`run_build` (`pipeline.py`) picks the path automatically based on how many
usable photos survive S0, and both converge on the same finishing stages
(clean → decimate to budget → UV/bake → export), so the CLI/dashboard command
is identical either way.

## Starting the dashboard

```bash
cd /Users/aditya/pixiecad
.venv/bin/pixiecad serve
```

Then open **http://127.0.0.1:8000**. From there:

1. Drop photos (drag-and-drop or file picker)
2. Set a name, target face count, optional real-world dimensions, "what is it"
3. Watch the job run — regime detection, stage log, warnings
4. Preview the model (three.js viewer), download the whole `.glb`
5. If "split into parts" was checked, download each named part separately

Options: `--port 8080`, `--host 0.0.0.0` (to reach it from another device on
your network), `--root <dir>` (where job workspaces are stored, default `jobs/`).

## CLI reference

```
pixiecad init       create a workspace + ObjectSpec (name, dimensions, face budget)
pixiecad ingest      S0 alone — score/dedupe/downscale a photo folder
pixiecad triage      S0.5 — coverage report before you commit to reconstructing
pixiecad sparse      S2a alone — camera poses + point cloud (needs pycolmap)
pixiecad build       the whole pipeline: photos in, budgeted .glb out
                        --backend fal|trellis-remote|fake   generative backend
                        --split --object "..."              also export parts
                        --host <ssh-alias>                   use a remote GPU for dense
pixiecad optimize    S4→S6 alone, given an existing dense mesh
pixiecad parts       split an existing mesh into named, budgeted parts
pixiecad drawings    S7 — orthographic SVGs from an existing mesh
pixiecad probe       check a host's GPU (local or --host <ssh-alias>)
pixiecad backends    which generative backends are configured right now
pixiecad cloud       gcloud status, running GPU instances, spend estimate
pixiecad serve       the web dashboard
```

Run any command with `--help` for its full option list.

## Remote GPU setup

Dense reconstruction (S2b) and the self-hosted generative backend need a CUDA
GPU with ≥16GB (dense) or ≥24GB (TRELLIS). This machine has neither, so those
stages run on a rented host over SSH:

```bash
./scripts/provision_gpu_vm.sh pixiecad-gpu asia-south1-a t4   # or: l4
pixiecad probe --host pixiecad-gpu.asia-south1-a.<project-id>
pixiecad build photos/ -w ws --host pixiecad-gpu.asia-south1-a.<project-id>
```

The script installs Docker + the NVIDIA container runtime and pulls the COLMAP
image; teardown is `gcloud compute instances delete <name> --zone=<zone>`.
Spot instances are used by default — a preempted job is a safe no-op re-run,
never lost work, because every stage is cached by content hash.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

154 tests, ~15 seconds, fully offline — no GPU, no network, no API keys.
