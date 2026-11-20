# PixieCAD

Turn photos of an object into a 3D model with a polygon budget you control.
Works two ways: **measure** a real object from many photos (photogrammetry),
or **generate** a plausible model from a few photos, real or AI-made.

## Install

Python 3.11+. Nothing here needs a GPU, a cloud account, or an API key to
install or to run the test suite.

```bash
git clone https://github.com/Ausevx/pixiecad.git
cd pixiecad
python3 -m venv .venv
.venv/bin/pip install -e ".[mesh,sfm,web,dev]"
```

The extras are separable, because most people do not want all of it:

| Extra | Brings | Needed for |
|---|---|---|
| *(base)* | numpy, pillow, opencv, pydantic, typer | ingest, scale, the CLI |
| `mesh` | trimesh, pyfqmr, xatlas, rtree, scipy, scikit-image | cleanup, decimate, UV/bake, drawings — the finishing half of the pipeline |
| `sfm` | pycolmap | sparse reconstruction (S2a), CPU-only |
| `web` | fastapi, uvicorn | the dashboard (`pixiecad serve`) |
| `bgremoval` | rembg | optional background removal on input photos |
| `dev` | pytest | the test suite |

Then, with no GPU and no network:

```bash
.venv/bin/pixiecad run images/ --label "demo" --faces 20000 --backend fake
```

`--backend fake` substitutes a placeholder mesh for the generative stage, so
the whole pipeline runs end to end locally. It proves the install; it does not
produce a real model.

## What you actually need

The pipeline splits cleanly into what runs on any laptop and what does not.

| Stage | Runs where | Requires |
|---|---|---|
| S0 ingest, S1 scale | Local | nothing beyond the base install |
| S2a sparse (poses + point cloud) | Local, CPU | `sfm` extra (pycolmap) |
| **S2b dense stereo** | **GPU host** | CUDA GPU ≥16 GB, Docker, COLMAP image |
| **S3 generative (image→mesh)** | **GPU host** | CUDA GPU ≥24 GB running Hunyuan3D or TRELLIS |
| S4 cleanup → S6 decimate/bake → S7 drawings | Local | `mesh` extra |
| S0.5 vision triage / part naming | Local | optional; works with **no** API key |
| Dashboard | Local | `web` extra |

The two GPU stages are the only hard external requirement, and they are reached
over plain SSH — see [Remote GPU setup](#remote-gpu-setup). Everything else is
your machine.

## Caveats, and what is specific to the author's setup

This project was built against one particular rig. Those choices are visible in
the repo, and none of them are load-bearing — but you should know which is
which before assuming something is required.

**GCP is a convenience, not a dependency.** `scripts/provision_gpu_vm.sh`,
`pixiecad cloud`, and the dashboard's compute drawer all shell out to `gcloud`,
because that is what the author rents GPUs from. None of it is on the critical
path. The pipeline's actual interface to a GPU is an **SSH alias** — anything
you can `ssh` into with a CUDA GPU and Docker works identically, whether that is
RunPod, Lambda, a university cluster, or a box under your desk:

```bash
pixiecad run images/ --host my-gpu-box     # any Host in ~/.ssh/config
```

Everything GCP-flavoured degrades rather than breaks. `pixiecad cloud` prints
`gcloud: not installed` and exits; the dashboard's compute panel reports
`available: false` and the rest of the UI works normally. `cloud.py` is imported
lazily at each call site — the `cloud` command and the dashboard's compute
endpoints — so nothing imports it unless you ask for it.

If you are not on GCP, ignore `scripts/*.sh` and the compute drawer entirely.

**Self-hosted models, by choice.** The generative backends are self-hosted
(Hunyuan3D, TRELLIS) rather than a hosted API. A fal.ai backend existed briefly
and was deliberately removed: it would have uploaded the user's photos off the
machine, which defeats the point. The consequence is that there is no zero-setup
generative path — you need a GPU host, or `--backend fake`.

**TRELLIS needs a HuggingFace token** for a gated dependency. Hunyuan3D does
not, which is why it is tried first. See [Remote GPU setup](#remote-gpu-setup).

**Generative output is not deterministic and not measured.** The photogrammetry
path reconstructs a real object and its dimensions mean something. The
generative path invents unseen surfaces; quality varies run to run, and the
output metadata flags which surfaces were invented. Do not treat a generative
result as a measurement.

**Developed on macOS against Linux GPU hosts.** The local half is pure Python
and portable; the remote half assumes a Linux host with Docker and the NVIDIA
container runtime. Windows is untested.

## Project structure

```
pixiecad/
├── PLAN.md                architecture notes / design log
├── pyproject.toml         deps, package config
├── scripts/
│   ├── provision_gpu_vm.sh   stand up a GCP GPU VM for dense/generative stages
│   ├── bake_image.sh          bake a custom GCP machine image with pre-pulled weights
│   ├── launch_from_image.sh   launch a VM from a baked image
│   ├── gcp_status.sh          check running instances and spend
│   ├── setup_hunyuan_vm.sh    set up Hunyuan3D on a GPU VM
│   ├── setup_hunyuan_texture.sh  set up Hunyuan texture worker
│   └── setup_sam_vm.sh        set up SAM segmentation worker
├── images/                sample F1 test set (generated, for pipeline testing)
├── runs/                  one folder per headless run (gitignored, see below)
├── src/pixiecad/
│   ├── cli.py             all commands (see below)
│   ├── session.py         one folder per run: input/ + work/ + output/, timestamped
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
│   │   ├── base.py            triangulate (self-hosted Hunyuan3D or TRELLIS
│   │   │                       on a GPU host, or "fake" for tests)
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
└── tests/                  172 tests, ~2s, no network/GPU required
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

## Headless run

One command, photos in and `.glb` out, with nothing to set up first:

```bash
.venv/bin/pixiecad run images/ --label "f1 car" --object "a Formula 1 race car" \
    --faces 20000 --split --host <gpu-ssh-alias>
```

Each run gets its own timestamped folder, so runs never overwrite each other
and an old one stays reproducible:

```
runs/
├── latest -> 20260802-181500-f1-car     symlink to the newest run
└── 20260802-181500-f1-car/
    ├── session.json     what was run: source photos, regime, backend, host
    ├── input/           the photos this run used (copied, not referenced)
    ├── work/            workspace + stage cache — scratch, safe to delete
    └── output/          the deliverables
        ├── model.glb        the whole object, at your face budget
        ├── parts/           one .glb per named part + manifest.json
        ├── drawings/        ortho SVGs, with --drawings
        └── build.json       per-stage status, timings, warnings
```

`--runs <dir>` puts the session folders somewhere else. Everything the run
needs is inside its own folder — `work/` can be deleted afterwards and
`output/` still stands on its own.

## Starting the dashboard

```bash
cd pixiecad
.venv/bin/pixiecad serve
```

Then open **http://127.0.0.1:8000**. From there:

1. Drop photos (drag-and-drop or file picker)
2. Set a name, target face count, optional real-world dimensions, "what is it"
3. Watch the job run — regime detection, stage log, warnings
4. Preview the model (three.js viewer), download the whole `.glb`
5. If "split into parts" was checked, download each named part separately

Each upload gets the same timestamped session folder as a headless run, so two
jobs can never share or overwrite each other's photos and meshes. `GET
/api/jobs/<id>` reports the folder in its `dir` and `output_dir` fields.

Options: `--port 8080`, `--host 0.0.0.0` (to reach it from another device on
your network), `--root <dir>` (where session folders are stored, default
`jobs/`).

### Themes

Six themes, reachable from the picker in the header: `dark` (amber on
near-black, the default), `light` (warm paper), `midnight` (cool blue-black),
`nord`, `sepia`, and `contrast` (WCAG AAA). A seventh choice, `system`, follows
the OS and keeps following it.

The whole visual system is CSS custom properties. `frontend/src/styles.css`
declares a `--px-*` set per theme, and an `@theme inline` block maps those to
Tailwind utilities — so a theme is defined entirely by its variables, and no
component carries a colour. The picker stores the choice in `localStorage` under
`pixiecad.theme` and stamps it on `<html>` as `data-theme`; `system` *removes*
the attribute, which is what lets the `prefers-color-scheme` media query take
over.

To add a theme:

1. Add a `:root[data-theme="<id>"]` block to `styles.css` setting **all 24**
   `--px-*` and `--field-*` variables plus `color-scheme`. Set every one — a
   missing variable silently inherits the dark theme's value and produces an
   unreadable mix rather than an obvious break.
2. Add an entry to `THEMES` in `frontend/src/lib/theme.ts` with its `id`,
   `label`, `base` (`light` or `dark`), and `swatch` — the swatch duplicates
   that theme's `--px-accent`, because the picker paints swatches for themes
   that are not currently applied and so cannot read the variable off `:root`.

Two `<canvas>` components (the background field and the logo) paint their own
pixels and cannot use the variables directly. Both observe the `data-theme`
attribute and re-read the palette on change, so they need no edit per theme —
but `--field-line` and `--field-point` must be real colour values, not
`transparent` or `currentColor`.

Contrast is a requirement, not a preference: ink clears 7:1 on both grounds and
accents clear 4.5:1, since the accent is used for text and links. The
`contrast` theme holds ink to 15:1.

## CLI reference

```
pixiecad run        headless end-to-end run into a fresh timestamped session folder
                        --label "..."      names the folder
                        --faces N          triangle budget
                        --split --object   also export named parts
                        --host <alias>     GPU host for dense/generative stages
                        --drawings         also write ortho SVGs
pixiecad init       create a workspace + ObjectSpec (name, dimensions, face budget)
pixiecad ingest      S0 alone — score/dedupe/downscale a photo folder
pixiecad triage      S0.5 — coverage report before you commit to reconstructing
pixiecad sparse      S2a alone — camera poses + point cloud (needs pycolmap)
pixiecad build       the whole pipeline: photos in, budgeted .glb out
                        --backend hunyuan-remote|trellis-remote|fake
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

Dense reconstruction (S2b) and the self-hosted generative backends need a CUDA
GPU with ≥16 GB (dense) or ≥24 GB (TRELLIS). If your machine has one, it is used
directly. If not, those stages run on another host over SSH.

**The interface is an SSH alias — nothing more.** Any host you can reach with a
CUDA GPU, Docker, and the NVIDIA container runtime works:

```bash
pixiecad probe --host my-gpu-box     # confirm the GPU is visible
pixiecad run images/ --host my-gpu-box
```

`my-gpu-box` is any `Host` entry in your `~/.ssh/config`. PixieCAD rsyncs the
work in, runs the stage in a container, and copies the result back. It does not
care who rents you the box.

### Provisioning on GCP (optional)

The author rents from GCP, so the repo carries scripts for it. **Skip this
section entirely if you have a GPU elsewhere** — nothing below is required.

```bash
# dense photogrammetry (S2b) — a T4 is plenty
./scripts/provision_gpu_vm.sh pixiecad-gpu asia-south1-a t4 colmap

# generative (S3) — needs an L4: the TRELLIS image is compiled for Ada (sm_89)
# only, so a T4 cannot launch its kernels no matter how much VRAM it reports
./scripts/provision_gpu_vm.sh pixiecad-l4 asia-south1-a l4 trellis

pixiecad run images/ --host pixiecad-l4.asia-south1-a.<project-id>
```

The script installs Docker + the NVIDIA container runtime and pre-pulls the
image for the chosen workload; teardown is
`gcloud compute instances delete <name> --zone=<zone>`. `PIXIECAD_GCP_PROJECT`
overrides the project id if it differs from your `gcloud config`.

**TRELLIS.2 needs a HuggingFace token.** It depends on
`facebook/dinov3-vitl16-pretrain-lvd1689m`, a gated repo, so before the first
generative run: accept the licence at
<https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>, create a
read token at <https://huggingface.co/settings/tokens>, then

```bash
export HF_TOKEN=hf_...
```

PixieCAD ships the token to the GPU host as a file inside the mounted cache,
never on a command line, so it does not show up in `ps` or `docker inspect`.
Without it the worker downloads 19 GB and loads for several minutes before
failing with a 401.

The TRELLIS worker is a service, not a batch job: PixieCAD starts it once,
waits for `/ready` (the first boot downloads the model), then posts views to
it over the host's own loopback — no public port is ever opened. The container
is left running between builds so later runs skip the multi-minute model load.
Spot instances are used by default — a preempted job is a safe no-op re-run,
never lost work, because every stage is cached by content hash.

## Tests

```bash
.venv/bin/python -m pytest -q
```

599 tests in about 90 seconds, fully offline — no GPU, no network, no API keys,
no credentials of any kind. That is the whole point of the `fake` backend and
the recorded fixtures: the suite must pass on a clean clone.

Two marker-gated tiers are excluded by default, configured in `pyproject.toml`:

```bash
.venv/bin/python -m pytest -q -m slow      # heavy: voxel repair, ffmpeg encodes
.venv/bin/python -m pytest -q -m ""        # everything, ~140s
```

`slow` holds the tests that build real video with `ffmpeg` or run voxel
dilate/fill/erode at two resolutions — genuinely heavy work rather than
incidentally sluggish tests. `network` holds anything needing the outside world;
nothing in the default run touches it.

The suite is not fast, and the honest reason is that a lot of it is integration
rather than unit work — the dashboard's endpoints and the mesh-repair
escalation ladder are tested by actually running them. If you are iterating on
one area, a targeted run is fast — `pytest -q tests/test_decimate.py
tests/test_cleanup.py` is 18 tests in 0.6s.

## License

MIT — see [LICENSE](LICENSE).

The sample photos in `images/` are AI-generated test fixtures, included so the
pipeline has something to chew on out of the box. They depict no real object and
carry no third-party rights.
