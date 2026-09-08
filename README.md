# t1-software

T1-LArTPC public data-challenge container (T1-FM foundation-models thrust).

One shared base image (Torch x JAX only) plus one overlay per detector
technology subgroup. Only the LArTPC subgroup exists so far.

```
t1-software/
+-- base/
|   +-- base.def          # Bootstrap: docker (nvidia/cuda) -> installs Torch + JAX
|   +-- versions.env       # CUDA_VARIANT, TORCH_VERSION, TORCH_INDEX_URL, JAX_VERSION
+-- subgroups/
|   +-- lartpc/
|       +-- lartpc.def      # Bootstrap: localimage (t1-base) -> clones + pins JAXTPC, pimm-data
|       +-- versions.env    # JAXTPC_COMMIT, PIMM_DATA_COMMIT
|       +-- tests/
|           +-- test_integration.py   # JAXTPC -> pimm-data HDF5 seam smoke test
+-- scripts/
|   +-- nightly-build.sh   # builds both images to $LSCRATCH, runs %test + integration test
+-- images/                 # gitignored symlink -> $LSCRATCH/t1-software-images
```

## Building locally

```bash
scripts/nightly-build.sh          # cuda12, the only variant built today
```

Requires `apptainer` with unprivileged `--fakeroot` builds (confirmed
working without a `/etc/subuid`/`/etc/subgid` entry on the nodes this has
been built on so far). No `docker`/`podman` daemon needed or used.

Builds land in `images/` (a symlink into `$LSCRATCH`, never git-tracked --
see `.gitignore`). Re-running is idempotent per tag (see below): the same
commit on the same day overwrites its own dated file, and always
re-points the `-latest.sif` symlinks.

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
  independent of the upstream pins (JAXTPC/pimm-data commits), which are
  recorded separately.

Two more files land next to each dated image in `images/`:

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
