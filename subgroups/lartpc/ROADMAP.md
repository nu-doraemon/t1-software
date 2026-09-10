# LArTPC subgroup roadmap

Sequencing for the `subgroups/lartpc` overlay, from today (JAXTPC + pimm-data
 + goop + DLPGenerator + edep-sim pinned, a synthetic-deposit integration
test plus a DLPGenerator -> edep-sim bomb-macro seam test, each vendored
package's own unit suite re-run in-image, cuda12, manually built, no CI) to
a v0.1 pre-release. Baseline models/evaluation are a separate container,
not tracked here.

- **Event generation** -- done: [DLPGenerator](https://github.com/DeepLearnPhysics/DLPGenerator)'s
  `ParticleBomb` generator drives
  [drinkingkazu/edep-sim](https://github.com/drinkingkazu/edep-sim) (a fork
  adding the `/generator/kinematics/set bomb` seam that links
  DLPGenerator's `ParticleBomb` directly) to produce H5DLP output;
  `DLPGENERATOR_COMMIT`/`EDEPSIM_COMMIT` in `versions.env`. Only a single
  low-multiplicity bomb-macro interaction is exercised by the seam test
  below so far -- not yet wired into the JAXTPC/pimm-data/goop
  charge-and-light chain.
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
- **Tests** -- JAXTPC's, pimm-data's, and goop's own unit-test suites re-run
  in-image as part of `%test` (all three self-skip GPU/asset-only cases),
  plus a DLPGenerator -> edep-sim bomb-macro -> H5DLP seam test
  (`tests/test_generator_edepsim.sh`); still pending: extend the
  integration test to the full generation-to-labeled chain (edep-sim ->
  labels -> JAXTPC charge -> goop light, disjoint seams today), and gate
  release on a real `--nv` run.
- **CI/CD** -- blocked on registry namespace + versioning-scheme decisions;
  once set, build/tag per-subgroup off its own pinned base tag, `cuda13`
  stays unverified (no GPU host yet) and isn't a v0.1 blocker.

Adding any of the above: one `<COMPONENT>_COMMIT` pin, one build stage, one
integration-test extension -- same pattern as JAXTPC/pimm-data already use.
