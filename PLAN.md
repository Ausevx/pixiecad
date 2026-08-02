# PixieCAD — Photos → Spec-Accurate 3D Models

## Goal

Take several photos of a real object from different angles and produce a detailed
3D model (`.glb`) with a **user-defined polygon budget**, optionally split into
**named, separately exportable parts** (wheels, chassis, …). Geometry from photos
is ground truth; generative completion of unseen regions is opt-in and clearly
flagged — never silent invention.

## Constraints (researched Aug 2026)

- **Host machine**: MacBook Air M3, 16 GB unified memory, no CUDA.
  - Local generative 3D (TRELLIS.2 MPS port needs 24 GB; SF3D wants 32 GB) is out.
  - `pycolmap` ships Apple Silicon wheels → local SfM works. Metal-SIFT fork exists.
  - Retopology / UV / bake stack (meshoptimizer, QuadriFlow/Instant Meshes, xatlas,
    trimesh) is CPU and runs fine locally.
- **LLMs cannot do geometry.** Their role is: photo triage feedback, part naming
  (VLM on cluster renders), QA of result renders vs input photos, and code-gen.
  Bulk VLM calls can be dispatched to agy/Gemini workers to save Anthropic quota.
- **Face-count control**: generative mesh models can't hit a target count
  (research consensus); a deterministic decimation/retopo stage can, exactly.
- **Key open models** (for the optional GPU/cloud stage):
  - TRELLIS.2-4B — MIT, single image → PBR mesh, 24 GB VRAM.
  - Hunyuan3D-Part (P3-SAM + X-Part) — part segmentation + part regeneration.
  - VGGT / π³ — pose-free feed-forward multi-view geometry (~7–12 GB VRAM).

## Pipeline (staged DAG, every stage cached & resumable)

```
photos ──▶ S0 ingest/optimize ──▶ S1 scale/dimensions ──▶ S2 geometry ──▶ S3 completion*
                                                                             │
export ◀── S6 retopo/UV/bake ◀── S5 parts* ◀── S4 cleanup ◀─────────────────┘
                                                              (* = optional)
```

- **S0 Ingest & input optimization** *(local, CPU — implemented first)*
  - EXIF read (focal length, orientation), auto-rotate.
  - Quality scoring: blur (variance of Laplacian), exposure (clipped-histogram),
    resolution floor. Reject/warn per photo with reasons.
  - Near-duplicate removal (perceptual hash) — fewer, better frames = less S2 work.
  - Downscale to working resolution (default 1600 px long edge).
  - Optional background removal (`rembg`, ONNX/CPU) for cleaner masks.
  - Report: usable views, angle coverage guess, actionable "take 3 more photos of
    the back-left" style feedback (VLM-assisted, optional).
- **S1 Scale & dimensions**
  - `ObjectSpec`: user-declared dimensions (`length=4.5m`, any 1–3 axes), units,
    target face budget, part export wishes.
  - ArUco marker auto-detection (OpenCV) for metric scale when present.
  - Applied as a similarity transform after S2; recorded in manifest.
- **S2 Geometry (backends behind one interface)**
  - `local-sfm`: pycolmap sparse → dense/mesh (slow, faithful, free, private).
  - `remote-cuda`: user's old NVIDIA laptop over SSH — rsync stage inputs, run
    COLMAP CUDA dense stereo (works on ancient GPUs, ~10–50× CPU) and, if
    `nvidia-smi` reports ≥6 GB, small neural models (TripoSR-class, rembg).
    VRAM/compute-capability detected at connect time gates model choice.
  - `cloud-gpu`: VGGT or TRELLIS.2 on rented GPU — user may have GCP credits;
    an L4 (24 GB, ~$0.70/hr, bursty ~5 min/object) covers all three GPU-only
    features (TRELLIS.2 completion, VGGT sparse-view, P3-SAM parts). Driven by
    the same SSH executor as `remote-cuda` — a GCP VM is "the laptop, but rented".
  - `api`: Tripo/Meshy multi-view — later, feature-flagged.
  - Android phone: not a compute node (cost/benefit is poor) — it's the capture
    client; S0 feedback loops back to it ("back-left under-covered, retake #7").
- **S3 Completion (opt-in)**: generative fill of unseen regions only; diff-masked
  so invented geometry is tagged in metadata.
- **S4 Cleanup**: weld/dedupe vertices, remove floaters & non-manifold edges,
  hole fill below threshold, normals fix (trimesh + custom ops).
- **S5 Parts (opt-in)**: P3-SAM-style segmentation (cloud) or geometric
  clustering (local fallback) → VLM names each part from turntable renders →
  user-editable part map → per-part meshes.
- **S6 Retopo / UV / bake — the polygon-budget contract**
  - Exact triangle budget: quadric decimation (meshoptimizer/pyfqmr) to N faces.
  - Optional quad remesh (Instant Meshes / QuadriFlow).
  - xatlas UV unwrap → bake normal/AO from the dense S4 mesh onto the low-poly.
  - Budget is per-model or per-part (e.g. wheels 800, chassis 5000).
- **Export**: `.glb` (assembly + per-part), manifest with provenance
  (real vs generated regions, scale source, quality report).

## Input regimes (user flow)

Same CLI/app regardless of input; S0 detects the regime, states it, and the
export manifest tags what was measured vs invented:

1. **Orbit capture** (20–40 photos): full photogrammetry, everything measured.
   Metric with dimensions/ArUco. The "spec-accurate" mode.
2. **Sparse random angles** (5–15 unposed photos): feed-forward geometry
   (VGGT-class, GPU) + generative completion for unseen surfaces — flagged.
3. **Single image**: pure TRELLIS.2 extrapolation; plausible asset, not a replica.

## Workload optimization

- Content-addressed stage cache (`workspace/<hash>/stageN/`) — re-runs only what
  changed; changing face budget re-runs S6 only, never reconstruction.
- Stages are subprocess-isolated so peak RAM stays bounded on 16 GB.
- Heavy stages declare `local | cloud | api` capability; scheduler picks the
  cheapest available that satisfies the request.

## Deliverable & phases

Python package `pixiecad` (library + Typer CLI) now; FastAPI + three.js viewer
web app after the pipeline produces good meshes (CLI-first was the agreed order).

1. **P1 (now)**: scaffold, S0 input optimization, `ObjectSpec`, workspace/cache.
2. **P2**: S2 local-sfm via pycolmap, S1 ArUco + dimension transform.
3. **P3**: S4 cleanup + S6 budgeted retopo/UV/bake → first photos→glb end-to-end.
4. **P4**: web app (upload → progress → three.js preview → slider → download).
5. **P5**: cloud backends (TRELLIS.2, VGGT), S3 completion, S5 parts + VLM naming.
