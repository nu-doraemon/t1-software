# LArTPC subgroup roadmap

Sequencing for the `subgroups/lartpc` overlay, from today (JAXTPC + pimm-data,
one synthetic-deposit smoke test, cuda12, manually built, no CI) to a v0.1
pre-release. Baseline models/evaluation are a separate container, not tracked
here.

- **Event generation** -- MPV/MPR + edep-sim; geometry is settled, only
  pending a co-worker's scripts/repos handoff.
- **Labels** -- [pysupera](https://github.com/DeepLearnPhysics/pysupera/tree/develop)
  groups edep-sim's reconstructed-particle point clouds into semantic
  partitions (shower/track/delta/Michel/low-energy-scatter); pin it and wire
  its output into the write/read seam alongside the deposits.
- **goop** -- [CIDeR-ML/goop](https://github.com/CIDeR-ML/goop) adds a
  light/PMT-readout branch (PyTorch) beside JAXTPC's charge/wire branch, both
  off the same labeled edep-sim steps; watch its Torch/CUDA pin, its own
  vendored `jaxtpc` submodule pin colliding with ours, and its multi-GB
  photon-library asset (don't bake into the image).
- **Tests** -- re-run each vendored package's own suite in-image, extend the
  integration test to the full labeled chain, gate release on a real `--nv`
  run.
- **CI/CD** -- blocked on registry namespace + versioning-scheme decisions;
  once set, build/tag per-subgroup off its own pinned base tag, `cuda13`
  stays unverified (no GPU host yet) and isn't a v0.1 blocker.

Adding any of the above: one `<COMPONENT>_COMMIT` pin, one build stage, one
integration-test extension -- same pattern as JAXTPC/pimm-data already use.
