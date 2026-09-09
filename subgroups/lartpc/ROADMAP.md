# LArTPC subgroup roadmap

Sequencing for the `subgroups/lartpc` overlay, from today (JAXTPC + pimm-data
 + goop pinned, one synthetic-deposit integration test plus each vendored
package's own unit suite re-run in-image, cuda12, manually built, no CI) to
a v0.1 pre-release. Baseline models/evaluation are a separate container,
not tracked here.

- **Event generation** -- MPV/MPR + edep-sim; geometry is settled, only
  pending a co-worker's scripts/repos handoff.
- **Labels** -- [pysupera](https://github.com/DeepLearnPhysics/pysupera/tree/develop)
  groups edep-sim's reconstructed-particle point clouds into semantic
  partitions (shower/track/delta/Michel/low-energy-scatter); pin it and wire
  its output into the write/read seam alongside the deposits.
- **goop** -- [CIDeR-ML/goop](https://github.com/CIDeR-ML/goop) adds a
  light/PMT-readout branch (PyTorch) beside JAXTPC's charge/wire branch, both
  off the same labeled edep-sim steps. Pinned and cloned into the image
  (`GOOP_COMMIT` in `versions.env`); its own vendored `jaxtpc` submodule is
  deliberately left uninitialized so its `tools.*` imports resolve against
  this image's single pinned `/opt/JAXTPC` checkout instead of a second,
  potentially skewed commit, and its multi-GB photon-library asset stays a
  host-side path read at runtime, never baked into the image. Not yet
  wired into the full labeled-chain integration test below -- so far only
  its own unit suite runs in-image.
- **Tests** -- JAXTPC's, pimm-data's, and goop's own unit-test suites now
  re-run in-image as part of `%test` (all three self-skip GPU/asset-only
  cases); still pending: extend the integration test to the full labeled
  chain (edep-sim -> labels -> JAXTPC charge -> goop light, one seam today),
  and gate release on a real `--nv` run.
- **CI/CD** -- blocked on registry namespace + versioning-scheme decisions;
  once set, build/tag per-subgroup off its own pinned base tag, `cuda13`
  stays unverified (no GPU host yet) and isn't a v0.1 blocker.

Adding any of the above: one `<COMPONENT>_COMMIT` pin, one build stage, one
integration-test extension -- same pattern as JAXTPC/pimm-data already use.
