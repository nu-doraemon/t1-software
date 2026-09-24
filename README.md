# t1-software

T1-LArTPC public data-challenge container (T1-FM foundation-models thrust).

One shared base image (Torch x JAX, plus a ROOT x GEANT4 conda-forge
layer used by event-generation subgroup overlays) plus one overlay per
detector technology subgroup. Only the LArTPC subgroup exists so far.

```
t1-software/
+-- base/
|   +-- base.def          # Bootstrap: docker (nvidia/cuda) -> installs Torch + JAX + ROOT/GEANT4 (conda-forge)
|   +-- versions.env       # CUDA_VARIANT, TORCH_VERSION, TORCH_INDEX_URL, JAX_VERSION, ROOT_VERSION, GEANT4_VERSION, ...
+-- subgroups/
|   +-- lartpc/
|       +-- lartpc.def      # Bootstrap: localimage (t1-base) -> clones + pins JAXTPC, pimm-data, goop, DLPGenerator, edep-sim
|       +-- versions.env    # JAXTPC_COMMIT, PIMM_DATA_COMMIT, GOOP_COMMIT, DLPGENERATOR_COMMIT, EDEPSIM_COMMIT
|       +-- tests/
|           +-- test_integration.py         # JAXTPC -> pimm-data HDF5 seam smoke test
|           +-- test_generator_edepsim.sh   # DLPGenerator -> edep-sim seam smoke test (bomb-macro -> H5DLP)
+-- scripts/
|   +-- nightly-build.sh   # builds both images, runs %test + integration test
+-- images/                 # gitignored default output dir (override via $T1_SOFTWARE_IMAGES_DIR)
```

## Building locally

```bash
scripts/nightly-build.sh          # cuda12, the only variant built today
```

Requires `apptainer` with unprivileged `--fakeroot` builds (confirmed
working without a `/etc/subuid`/`/etc/subgid` entry on the nodes this has
been built on so far). No `docker`/`podman` daemon needed or used.
Apptainer's build scratch/cache, and the finished `.sif` images
themselves, are staged under `$LSCRATCH` (required) and only moved into
the images dir below once both builds and the `%test` step succeed --
building a multi-GB image under a small default `/tmp` fails or
thrashes, and writing the finished multi-GB `.sif` directly to a
shared/NFS images dir mid-build is much slower than one local build plus
one final move.

Final images land in `images/` next to this repo by default (gitignored
-- see `.gitignore`). Set `$T1_SOFTWARE_IMAGES_DIR` before invoking the
script to redirect them elsewhere -- e.g. a workspace embedding this repo
under its own `src/` typically points it at that workspace's own image
store rather than accepting the default. Re-running is idempotent per tag
(see below): the same commit on the same day overwrites its own dated
file, and always re-points the `-latest.sif` symlinks.

## Runtime environment

The image has one Python: conda-forge CPython 3.11 at `/opt/conda`,
carrying Torch, JAX, PyROOT, and every subgroup package. It is active in
every `apptainer exec/run/shell` with no setup step -- `python3`,
`root`, `dlpgen`, `edep-sim`, and the Geant4 data variables all work
directly.

To *compile* against ROOT/Geant4 (e.g. rebuilding DLPGenerator or
edep-sim yourself), first run `source /opt/setup-g4root.sh`. It adds only
conda's compiler settings (`CC`/`CXX`/`CFLAGS`/`LDFLAGS`, matching the
GCC ROOT was built with); these are kept out of the default environment
so unrelated builds never pick them up.

## Image versioning

Every image `nightly-build.sh` produces is named:

```
<layer>-<cuda-variant>-<YYYYMMDD>-<git-sha>[-dirty].sif
```

- `layer`: `base`, or a subgroup name (`lartpc`; future subgroups add
  their own).
- `cuda-variant`: `cuda12` today; `cuda13` once a driver >=580.65.06 host
  exists to build and verify it.
- `YYYYMMDD`: UTC build date.
- `git-sha`: this repo's short (8-char) commit at build time, suffixed
  `-dirty` if the working tree had uncommitted changes. This ties an
  image back to the exact `.def`/`versions.env` recipe that produced it,
  independent of the upstream pins (JAXTPC/pimm-data/goop commits), which are
  recorded separately.

Two more files land next to each dated image in the output directory:

- `<name>.manifest.json` -- build timestamp, cuda variant, this repo's git
  commit, and the layer's upstream pins (from its `versions.env`).
- `<layer>-<cuda-variant>-latest.sif` -- a symlink to the newest dated
  build for that layer+variant. **This is the pointer to follow for "the
  latest nightly build"**; scripts and downstream consumers should target
  it (e.g. `images/t1-lartpc-cuda12-latest.sif`) rather than a dated
  filename, unless they specifically need to pin an exact historical
  build. The symlink is updated atomically (write-then-rename) on every
  successful run.

Dated images accumulate across runs (nothing here prunes old builds);
that is a deliberate simplification for now, not a limitation.

## Adding a new detector-technology subgroup

Copy `subgroups/lartpc/` as a template: one `<name>.def` bootstrapping
`Bootstrap: localimage / From: {{ BASE_SIF }}`, one `versions.env` for its
own third-party pins, and a `tests/` directory with at least one
integration smoke test proving the subgroup's own data-pipeline seam.
Wire it into `scripts/nightly-build.sh`'s subgroup loop.
