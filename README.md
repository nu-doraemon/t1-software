# t1-software

T1 public data-challenge containers (T1-FM foundation-models thrust):
t1-lartpc (LArTPC) and t1-wc (water Cherenkov).

One shared base image (Torch x JAX, plus a ROOT x GEANT4 conda-forge
layer used by event-generation subgroup overlays) plus one overlay per
detector technology subgroup: `lartpc` and `wc`. Both overlays build
`FROM` the same base image in the same run. Every image is built once,
as an OCI (Docker/Podman) image; the Apptainer `.sif` is converted from
that same image.

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
|       +-- versions.env    # <PKG>_REPO clone URL + <PKG>_COMMIT pin per package (JAXTPC, pimm-data, goop, chroma-lar, TriChroma, photonlib, DLPGenerator, edep-sim, pysupera, dprod), TRICHROMA_TRITON_VERSION
|       +-- build/build-generators.sh   # DLPGenerator + edep-sim build, run inside the image build
|       +-- tests/
|           +-- image-test.sh               # all in-image checks (imports, seams, vendored unit suites)
|           +-- test_integration.py         # JAXTPC -> pimm-data HDF5 seam smoke test
|           +-- test_generator_edepsim.sh   # DLPGenerator -> edep-sim seam smoke test (bomb-macro -> H5DLP)
|           +-- dprod_site_image.yaml       # dprod site config: in-image checkouts, local scheduler, no nested container
|   +-- wc/
|       +-- Containerfile   # FROM t1-base -> LUCiD (editable, base packages pinned) + PhotonSim on its own Geant4 11.3 prefix
|       +-- versions.env    # LUCID_*/PHOTONSIM_* repo + commit pins, PHOTONSIM_* toolchain pins (Geant4, ROOT, GCC)
|       +-- build/build-photonsim.sh    # PhotonSim build against /opt/photonsim-g4 only, Geant4 data links, wrapper install
|       +-- runtime/photonsim           # $PHOTONSIM_BIN: PhotonSim with the sidecar's Geant4 data/ROOTSYS
|       +-- tests/
|           +-- image-test.sh               # base packages unchanged, imports, seam, LUCiD unit suite
|           +-- test_photonsim_lucid.sh     # PhotonSim -> LUCiD -> HDF5 seam (lucid-run-job, 3 muons)
+-- scripts/
|   +-- build.sh           # podman build of base + every subgroup overlay, tests, .sif conversion, dprod full-chain test
|   +-- dprod-pipeline-test.sh  # dprod-install-test inside a built lartpc image (GPU node)
|   +-- tests/             # fast tests of build.sh with fake podman/apptainer
+-- sites/s3df/            # nightly HEAD builds on S3DF (cron + Slurm + build ledger + dashboard)
+-- images/                # gitignored default output dir (override with --images-dir)
```

## Building

```bash
scripts/build.sh                  # cuda12, base + every subgroup, the pins in versions.env
scripts/build.sh --subgroup wc    # base + t1-wc only
scripts/build.sh --help           # every option and its default
```

| Option | Default |
|---|---|
| `--variant NAME` | `cuda12` (the only variant built today) |
| `--work-dir DIR` | `$LSCRATCH/t1-software-build` if `$LSCRATCH` is set, else `${TMPDIR:-/tmp}/t1-software-build` |
| `--images-dir DIR` | `$T1_SOFTWARE_IMAGES_DIR`, else `images/` next to this repo |
| `--cache-dir DIR` | `$T1_SOFTWARE_CACHE_DIR`; unset means node-local cache only |
| `--pin KEY=VALUE` | none; overrides one `versions.env` key for this build; repeatable |
| `--subgroup NAME` | all (`lartpc`, `wc`); repeatable; builds only the named overlays |
| `--tag TAG` | `<YYYYMMDD>-<git-sha8>[-dirty]` |
| `--no-sif` | off; skips the `.sif`, the apptainer tests and the pipeline test |
| `--no-base-archive` | off; skips the t1-base `.oci.tar` and its `-latest` link; the base is still built, tested and recorded in its manifest |
| `--no-pipeline-test` | off; on when `T1_SKIP_PIPELINE_TEST=1` |
| `--gpu auto\|on\|off` | `auto`: apptainer tests get `--nv` when `nvidia-smi` finds a GPU |
| `--squashfs-mem SIZE` | `2G` |

`--pin` builds any commit set without editing a file. Example:
`scripts/build.sh --pin GOOP_COMMIT=<sha>` builds goop at `<sha>`. A key
that is in no `versions.env`, or in more than one, is an error, and the
script stops before it starts podman. A key of a subgroup that is not
selected is accepted and has no effect. The manifests record every pin
that was used.

Requires rootless `podman` and `apptainer`. No `/etc/subuid` or
`/etc/subgid` entry is needed: the script sets
`overlay.ignore_chown_errors=true` (a single-UID user namespace cannot
`chown` to other users, so every file in the image is owned by the build
user), and the Containerfiles run apt's sandbox as root.

Podman storage, temp files, and the staged outputs live in the work dir.
Put it on a node-local disk: rootless overlay needs a local filesystem,
and writing multi-GB files straight to a shared/NFS images dir is much
slower than one local write plus one move. Plan for about 110 GB there:
about 30 GB of layers plus the three archives and two `.sif`s (about
70 GB together; the t1-wc pair alone is about 30 GB). To clear the
storage, run:

```bash
podman --root <work-dir>/podman/root \
       --runroot <work-dir>/podman/run system reset -f
```

Plain `rm -rf` fails on busy overlay mounts.

Final images land in `images/` next to this repo by default (gitignored
-- see `.gitignore`). Pass `--images-dir` (or set
`$T1_SOFTWARE_IMAGES_DIR`) to redirect them elsewhere -- e.g. a workspace
embedding this repo under its own `src/` typically points it at its own
image store. Re-running is idempotent per tag (see below): the same
commit on the same day overwrites its own dated files, and always
re-points the `-latest` symlinks.

Memory: request at least 16 GB for the job. JAXTPC's unit suite alone
peaks at about 10.7 GB RSS, and the apptainer test run is charged to
the job's cgroup. An 8 GB allocation OOM-kills it. (Rootless podman
containers run outside the Slurm job cgroup, so the podman test run
passes even there.)

`scripts/tests/` tests the option, pin, tag, file-set and manifest logic
in seconds, with fake `podman` and `apptainer` on `PATH`:

```bash
python3 -m unittest discover -s scripts/tests -v
```

### Layer cache

Each pinned package is its own build step, placed after its own `ARG`.
A rebuild reuses every step above the first changed pin. By default the
cache lives in the podman storage in the work dir, so only the next
build on the same node reuses it.

To share the cache across nodes, pass `--cache-dir` (or set
`$T1_SOFTWARE_CACHE_DIR`) to a persistent directory on shared storage:

```bash
scripts/build.sh --cache-dir /path/to/shared/t1-software-cache
```

During the builds, the script runs a registry container
(`registry:2.8.3`, loopback only, random port) whose blob store is
`<cache-dir>/registry`, and builds with `--cache-from` /
`--cache-to` against it. No external registry or account is involved.

- **Size:** one full set of layers is about 19 GB (6.4 GB of it is the
  `nvidia/cuda` base). Each rebuild adds the layers it changed: about
  0.1-0.5 GB for a subgroup pin, about 5 GB for Torch/JAX, 7-8 GB for
  ROOT/Geant4/Python. Nothing is pruned; plan for about 50 GB and delete
  the directory when it exceeds that. The next build repopulates it.
- **Concurrency:** one build at a time. A second build fails fast on
  `<cache-dir>/.lock`. The lock is released once the builds
  finish, before the tests start.
- **Not cached:** the `FROM nvidia/cuda` base is pulled from Docker Hub
  on a node that does not have it yet.

### Tests run by the build

- `base/tests/image-test.sh` and `subgroups/<sub>/tests/image-test.sh`
  of every built subgroup. They run twice: under podman (CPU) and under
  apptainer on the converted `.sif` (`--nv` when a GPU is present); the
  base checks run on each subgroup's `.sif` too. No GPU is wired into
  podman.
  - lartpc: imports, the JAXTPC -> pimm-data and DLPGenerator ->
    edep-sim seam tests, and the full unit suites of JAXTPC, pimm-data,
    goop and pysupera.
  - wc: the base's Python packages, Geant4 and ROOT are the ones the
    base shipped; LUCiD/flax/optax import; LUCiD's provenance (its own
    commit and PhotonSim's commit, tag and Geant4, from `/opt/VERSIONS`)
    resolves; the PhotonSim -> LUCiD -> HDF5 seam (`lucid-run-job` on a
    three-muon smoke config with `--detector SK_WAND`, under a minute);
    and LUCiD's default `pytest tests/` collection (about 5-6 min on 8
    cores; LUCiD's conftest forces JAX onto the CPU, and tests needing
    LUCiD's downloaded SIREN data skip).
- `scripts/dprod-pipeline-test.sh [--work-dir DIR] <lartpc.sif>` (only when lartpc is built): dprod's
  own `dprod-install-test` inside the image -- one job through edep-sim ->
  JAXTPC (wire) -> pysupera -> JAXTPC (pixel) -> pysupera with dprod's
  local scheduler (no slurm), then dprod's event-id and provenance-chain
  checks. It needs a GPU (`--nv`); `build.sh` runs it before any image
  leaves the work dir and fails on a GPU-less node unless
  `--no-pipeline-test` (or `T1_SKIP_PIPELINE_TEST=1`). It also runs
  standalone against any built image.

## Nightly builds

`sites/s3df/` builds the HEAD of every tracked package once a day on
S3DF, keeps a ledger of every build, and renders a static dashboard. It
calls `build.sh` with `--pin` and `--tag` and never edits this repo. See
`sites/s3df/README.md`; another site can copy that directory.

Dashboard of the S3DF builds:
<https://s3df.slac.stanford.edu/data/neutrino/t1-software/index.html>

## Using the images

```bash
apptainer exec --nv images/t1-lartpc-cuda12-latest.sif python3 ...
apptainer exec --nv images/t1-wc-cuda12-latest.sif lucid-run-job --config ... --detector SK_WAND --job-id 1 --output-dir out/
podman load -i images/t1-lartpc-cuda12-latest.oci.tar     # or: docker load -i
podman run --rm localhost/t1-lartpc-cuda12:<tag> python3 ...
```

### WC subgroup

- **LUCiD** is an editable checkout at `/opt/LUCiD` (it reads `config/`
  and `data/` next to its package), installed into `/opt/conda` with the
  base's Torch/JAX/NumPy/SciPy/h5py held at their base versions. LUCiD
  supports only `jax>=0.6.2,<0.7`; the base pins JAX 0.6.2.
  Its PMT overlap tables for the shipped `*_WAND` detectors are built
  into the image.
- **`/opt/VERSIONS`** records the LUCiD and PhotonSim commits and tags
  and PhotonSim's Geant4 version; LUCiD copies them into every dataset
  batch it writes.
- **PhotonSim** runs from `$PHOTONSIM_BIN` (`/opt/PhotonSim/bin/photonsim`),
  which `lucid-run-job` reads. PhotonSim needs Geant4 >= 11.0, while the
  base stays on Geant4 10.7 for the LArTPC overlay's edep-sim fork, so
  PhotonSim is built against its own conda prefix `/opt/photonsim-g4`
  (Geant4 11.3, ROOT, GCC). The wrapper gives only the PhotonSim process
  that prefix's Geant4 data, `ROOTSYS` and sysroot; the image's ENV,
  `geant4-config` and `root` stay the base's.
- **Not in the image:** LUCiD's SIREN example data (run
  `python scripts/download_data.py` from a writable LUCiD checkout on
  the host), GENIE (`primary_source: genie` configs do not run; LUCiD
  ships none), and `snewpy` (supernova config only).
- **Licenses:** LUCiD is Apache-2.0. PhotonSim ships no license file at
  the pinned commit; settle that before redistributing the image outside
  the collaboration.

## Runtime environment

Each image has one Python: conda-forge CPython 3.11 at `/opt/conda`,
carrying Torch, JAX, PyROOT, and that subgroup's packages. It is active
with no setup step -- `python3`, `root`, the Geant4 data variables, and
the subgroup's tools (`dlpgen` and `edep-sim` in t1-lartpc,
`lucid-run-job` and `$PHOTONSIM_BIN` in t1-wc) all work directly.

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

Every file `build.sh` produces is named (with the default tag):

```
<layer>-<cuda-variant>-<YYYYMMDD>-<git-sha>[-dirty].{oci.tar,sif,manifest.json}
```

- `layer`: `base`, or a subgroup name (`lartpc`, `wc`).
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
| `t1-<sub>-<variant>-<tag>.oci.tar` | per subgroup (`lartpc`, `wc`); `podman load -i` / `docker load -i`; contains every layer, base included |
| `t1-<sub>-<variant>-<tag>.sif` | apptainer image converted from that OCI image |
| `t1-base-<variant>-<tag>.oci.tar` | the base alone, to build an overlay on another node without rebuilding base |
| `<layer>-<variant>-<tag>.manifest.json` | build timestamp, cuda variant, this repo's git commit, and the layer's upstream pins |

`t1-<sub>-<variant>-latest.sif`, `t1-<sub>-<variant>-latest.oci.tar`
and `t1-base-<variant>-latest.oci.tar` are symlinks to the newest dated
build in the same images dir. **These are the pointers to follow for
"the latest build"**; scripts and downstream consumers should target
them (e.g. `images/t1-lartpc-cuda12-latest.sif`) rather than a dated
filename, unless they specifically need to pin an exact historical
build. The symlinks are updated atomically (write-then-rename) on every
successful run.

`build.sh` itself prunes nothing; dated images accumulate. The nightly
layer in `sites/s3df/` prunes its own old images.

## Adding a new detector-technology subgroup

Copy `subgroups/wc/` (the smaller one) as a template: one `Containerfile`
starting `ARG BASE_IMAGE` / `FROM ${BASE_IMAGE}`, one `versions.env` for
its own third-party pins (key names unique across every `versions.env`),
its runtime paths as `ENV` lines ending with `RUN
/opt/t1-runtime/check-env.sh`, and a `tests/` directory with an
`image-test.sh` (copied to `/opt/t1-<name>-tests/`) plus at least one
integration smoke test proving the subgroup's own data-pipeline seam.
Add the name to `SUBGROUPS_ALL` in `scripts/build.sh` and to
`SUBGROUPS` in `sites/s3df/nightly.py`.
