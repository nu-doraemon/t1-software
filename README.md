# t1-software

T1-LArTPC public data-challenge container (T1-FM foundation-models thrust).

One shared base image (Torch x JAX, plus a ROOT x GEANT4 conda-forge
layer used by event-generation subgroup overlays) plus one overlay per
detector technology subgroup. Only the LArTPC subgroup exists so far.
Every image is built once, as an OCI (Docker/Podman) image; the
Apptainer `.sif` is converted from that same image.

```
t1-software/
+-- base/
|   +-- Containerfile      # FROM nvidia/cuda -> Torch + JAX + ROOT/GEANT4 (conda-forge)
|   +-- versions.env       # CUDA_VARIANT, TORCH_VERSION, TORCH_INDEX_URL, JAX_VERSION, ROOT_VERSION, GEANT4_VERSION, ...
|   +-- runtime/           # build-time env helpers (g4data links, ENV drift check), setup-g4root.sh
|   +-- tests/image-test.sh
+-- subgroups/
|   +-- lartpc/
|       +-- Containerfile   # FROM t1-base -> clones + pins JAXTPC, pimm-data, goop (+ chroma-lar, TriChroma, photonlib, Triton override), DLPGenerator, edep-sim, pysupera, dprod
|       +-- versions.env    # JAXTPC_COMMIT, PIMM_DATA_COMMIT, GOOP_COMMIT, CHROMA_LAR_COMMIT, TRICHROMA_COMMIT, PHOTONLIB_COMMIT, TRICHROMA_TRITON_VERSION, DLPGENERATOR_COMMIT, EDEPSIM_COMMIT, PYSUPERA_COMMIT, DPROD_COMMIT
|       +-- build/build-generators.sh   # DLPGenerator + edep-sim build, run inside the image build
|       +-- tests/
|           +-- image-test.sh               # all in-image checks (imports, seams, vendored unit suites)
|           +-- test_integration.py         # JAXTPC -> pimm-data HDF5 seam smoke test
|           +-- test_generator_edepsim.sh   # DLPGenerator -> edep-sim seam smoke test (bomb-macro -> H5DLP)
|           +-- dprod_site_image.yaml       # dprod site config: in-image checkouts, local scheduler, no nested container
+-- scripts/
|   +-- nightly-build.sh   # podman build of both layers, tests, .sif conversion, dprod full-chain test
|   +-- dprod-pipeline-test.sh  # dprod-install-test inside a built lartpc image (GPU node)
+-- images/                 # gitignored default output dir (override via $T1_SOFTWARE_IMAGES_DIR)
```

## Building locally

```bash
scripts/nightly-build.sh          # cuda12, the only variant built today
```

Requires rootless `podman` and `apptainer`. No `/etc/subuid` or
`/etc/subgid` entry is needed: the script sets
`overlay.ignore_chown_errors=true` (a single-UID user namespace cannot
`chown` to other users, so every file in the image is owned by the build
user), and the Containerfiles run apt's sandbox as root.

Podman storage, temp files, and the staged outputs live under
`$LSCRATCH/t1-software-build/` (required). Rootless overlay needs a local
filesystem, and writing multi-GB files straight to a shared/NFS images
dir is much slower than one local write plus one move. Plan for about
80 GB there: about 25 GB of layers plus the archives and the `.sif`. To
clear the storage, run:

```bash
podman --root $LSCRATCH/t1-software-build/podman/root \
       --runroot $LSCRATCH/t1-software-build/podman/run system reset -f
```

Plain `rm -rf` fails on busy overlay mounts.

Final images land in `images/` next to this repo by default (gitignored
-- see `.gitignore`). Set `$T1_SOFTWARE_IMAGES_DIR` before invoking the
script to redirect them elsewhere -- e.g. a workspace embedding this repo
under its own `src/` typically points it at that workspace's own image
store rather than accepting the default. Re-running is idempotent per tag
(see below): the same commit on the same day overwrites its own dated
files, and always re-points the `-latest` symlinks.

Memory: request at least 16 GB for the job. JAXTPC's unit suite alone
peaks at about 10.7 GB RSS, and the apptainer test run is charged to
the job's cgroup. An 8 GB allocation OOM-kills it. (Rootless podman
containers run outside the Slurm job cgroup, so the podman test run
passes even there.)

### Layer cache

Each pinned package is its own build step, placed after its own `ARG`.
A rebuild reuses every step above the first changed pin. By default the
cache lives in the podman storage on `$LSCRATCH`, so only the next build
on the same node reuses it.

To share the cache across nodes, point `$T1_SOFTWARE_CACHE_DIR` at a
persistent directory on shared storage:

```bash
T1_SOFTWARE_CACHE_DIR=/path/to/shared/t1-software-cache scripts/nightly-build.sh
```

During the two builds, the script runs a registry container
(`registry:2.8.3`, loopback only, random port) whose blob store is
`$T1_SOFTWARE_CACHE_DIR/registry`, and builds with `--cache-from` /
`--cache-to` against it. No external registry or account is involved.

- **Size:** one full set of layers is about 14 GB (6.4 GB of it is the
  `nvidia/cuda` base). Each rebuild adds the layers it changed: about
  0.1-0.5 GB for a subgroup pin, about 5 GB for Torch/JAX, 7-8 GB for
  ROOT/Geant4/Python. Nothing is pruned; plan for about 50 GB and delete
  the directory when it exceeds that. The next build repopulates it.
- **Concurrency:** one build at a time. A second build fails fast on
  `$T1_SOFTWARE_CACHE_DIR/.lock`. The lock is released once the builds
  finish, before the tests start.
- **Not cached:** the `FROM nvidia/cuda` base is pulled from Docker Hub
  on a node that does not have it yet.

### Tests run by the build

- `base/tests/image-test.sh` and `subgroups/lartpc/tests/image-test.sh`:
  imports, the JAXTPC -> pimm-data and DLPGenerator -> edep-sim seam
  tests, and the full unit suites of JAXTPC, pimm-data, goop and
  pysupera. They run twice: under podman (CPU) and under apptainer on
  the converted `.sif` (`--nv` when a GPU is present). No GPU is wired
  into podman.
- `scripts/dprod-pipeline-test.sh <lartpc.sif>`: dprod's own
  `dprod-install-test` inside the image -- one job through edep-sim ->
  JAXTPC (wire) -> pysupera -> JAXTPC (pixel) -> pysupera with dprod's
  local scheduler (no slurm), then dprod's event-id and provenance-chain
  checks. It needs a GPU (`--nv`); `nightly-build.sh` runs it before any
  image leaves scratch and fails on a GPU-less node unless
  `T1_SKIP_PIPELINE_TEST=1`. It also runs standalone against any built image.

## Using the images

```bash
apptainer exec --nv images/t1-lartpc-cuda12-latest.sif python3 ...
podman load -i images/t1-lartpc-cuda12-latest.oci.tar     # or: docker load -i
podman run --rm localhost/t1-lartpc-cuda12:<tag> python3 ...
```

## Runtime environment

The image has one Python: conda-forge CPython 3.11 at `/opt/conda`,
carrying Torch, JAX, PyROOT, and every subgroup package. It is active
with no setup step -- `python3`, `root`, `dlpgen`, `edep-sim`, and the
Geant4 data variables all work directly.

The whole runtime environment is static `ENV` in the Containerfiles; no
entrypoint or sourced script is involved. Every launch path sees it:
`docker/podman run` and `exec`, an overridden `--entrypoint`, and
`apptainer exec/run/shell` (Apptainer imports OCI `ENV`).

The values mirror what conda's `activate.d` hooks set. The Geant4
dataset variables point at version-free symlinks, `/opt/g4data/<VAR>`,
so a geant4 pin bump does not change the `ENV` lines. The last build
step of every Containerfile runs `/opt/t1-runtime/check-env.sh`: it
reruns the hooks in a clean shell and fails the build if `ENV` misses
any variable they set. On failure, add the reported variable to the
Containerfile's `ENV`.

To *compile* against ROOT/Geant4 (e.g. rebuilding DLPGenerator or
edep-sim yourself), first run `source /opt/setup-g4root.sh`. It adds only
conda's compiler settings (`CC`/`CXX`/`CFLAGS`/`LDFLAGS`, matching the
GCC ROOT was built with); these are kept out of the default environment
so unrelated builds never pick them up.

## Image versioning

Every file `nightly-build.sh` produces is named:

```
<layer>-<cuda-variant>-<YYYYMMDD>-<git-sha>[-dirty].{oci.tar,sif,manifest.json}
```

- `layer`: `base`, or a subgroup name (`lartpc`; future subgroups add
  their own).
- `cuda-variant`: `cuda12` today; `cuda13` once a driver >=580.65.06 host
  exists to build and verify it.
- `YYYYMMDD`: UTC build date.
- `git-sha`: this repo's short (8-char) commit at build time, suffixed
  `-dirty` if the working tree had uncommitted changes. This ties an
  image back to the exact `Containerfile`/`versions.env` recipe that
  produced it, independent of the upstream pins (JAXTPC/pimm-data/goop
  commits), which are recorded separately.

Per build:

| File | Use |
|---|---|
| `t1-lartpc-<variant>-<tag>.oci.tar` | `podman load -i` / `docker load -i`; contains every layer, base included |
| `t1-lartpc-<variant>-<tag>.sif` | apptainer image converted from that OCI image |
| `t1-base-<variant>-<tag>.oci.tar` | the base alone, to build an overlay on another node without rebuilding base |
| `<layer>-<variant>-<tag>.manifest.json` | build timestamp, cuda variant, this repo's git commit, and the layer's upstream pins |

`t1-lartpc-<variant>-latest.sif`, `t1-lartpc-<variant>-latest.oci.tar`
and `t1-base-<variant>-latest.oci.tar` are symlinks to the newest dated
build. **These are the pointers to follow for "the latest nightly
build"**; scripts and downstream consumers should target them (e.g.
`images/t1-lartpc-cuda12-latest.sif`) rather than a dated filename,
unless they specifically need to pin an exact historical build. The
symlinks are updated atomically (write-then-rename) on every successful
run.

Dated images accumulate across runs (nothing here prunes old builds);
that is a deliberate simplification for now, not a limitation.

## Adding a new detector-technology subgroup

Copy `subgroups/lartpc/` as a template: one `Containerfile` starting
`ARG BASE_IMAGE` / `FROM ${BASE_IMAGE}`, one `versions.env` for its own
third-party pins, its runtime paths as `ENV` lines ending with `RUN
/opt/t1-runtime/check-env.sh`, and a `tests/` directory with an
`image-test.sh` plus
at least one integration smoke test proving the subgroup's own
data-pipeline seam. Wire it into `scripts/nightly-build.sh`.
