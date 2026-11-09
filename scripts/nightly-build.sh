#!/usr/bin/env bash
# Nightly (or on-demand) build of the T1-LArTPC container stack.
#
#   scripts/nightly-build.sh [cuda12]   # default: cuda12
#
# Builds base/Containerfile and subgroups/lartpc/Containerfile with
# rootless podman, then derives the apptainer .sif from the same OCI
# image. Outputs, in $T1_SOFTWARE_IMAGES_DIR (default ./images next to
# this repo; this repo never assumes a shared location):
#   t1-base-<variant>-<tag>.oci.tar     base only; `podman load` it to
#                                       build an overlay elsewhere
#   t1-lartpc-<variant>-<tag>.oci.tar   `podman load -i` / `docker load -i`
#   t1-lartpc-<variant>-<tag>.sif       apptainer image of the same OCI image
#   <layer>-<variant>-<tag>.manifest.json, and -latest symlinks per file kind
#
# Rootless podman without a /etc/subuid entry maps a single UID, so:
#   - overlay.ignore_chown_errors=true lets layers that chown to other
#     users unpack (every file ends up owned by the build user);
#   - the Containerfiles run apt's sandbox as root.
# Podman storage, temp files and staged outputs all live on node-local
# $LSCRATCH: rootless overlay needs a local filesystem, and writing
# multi-GB files straight to a shared/NFS images dir is much slower than
# one local write plus one move. The layer cache in that storage is
# reused by the next build on the same node only.
#
# Shared layer cache (optional): set $T1_SOFTWARE_CACHE_DIR to a
# persistent directory on shared storage. For the two builds, the script
# starts a registry container (REGISTRY_IMAGE, below) on 127.0.0.1 with
# its blob store in $T1_SOFTWARE_CACHE_DIR/registry, and builds with
# --cache-from/--cache-to against it. A build on any node then reuses
# every step whose inputs are unchanged. One build at a time may use the
# cache: a second one fails fast on $T1_SOFTWARE_CACHE_DIR/.lock. The
# cache is disposable -- delete the directory to reclaim space; the next
# build repopulates it. Unset, the script uses the node-local cache only.
#
# Tests: the image-test.sh scripts run under podman (CPU) and under
# apptainer (--nv when a GPU is present); both see the image's static
# ENV with no setup step. Then dprod-pipeline-test.sh runs
# on the .sif. Nothing leaves $LSCRATCH unless all of them pass.
#
# Versioning: see README.md "Image versioning". Tag is
# <date>-<git-sha>[-dirty]; a JSON manifest next to each layer records the
# pins that produced it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Absolute before the cd below: a relative path is meant from the
# caller's cwd, and podman's -v reads a relative source as a volume name.
if [[ -n "${T1_SOFTWARE_CACHE_DIR:-}" ]]; then
    T1_SOFTWARE_CACHE_DIR="$(realpath -m "$T1_SOFTWARE_CACHE_DIR")"
fi
cd "$REPO_ROOT"

VARIANT="${1:-cuda12}"
case "$VARIANT" in
    cuda12) ;;
    cuda13) echo "nightly-build: cuda13 not built by this script yet (no verified >=580.65.06 driver host to build/test it on)." >&2; exit 1 ;;
    *) echo "nightly-build: unknown variant '$VARIANT' (expected cuda12)" >&2; exit 1 ;;
esac

: "${LSCRATCH:?nightly-build: \$LSCRATCH is not set (needs a batch/interactive allocation that provides it)}"
BUILD_ROOT="$LSCRATCH/t1-software-build"
export TMPDIR="$BUILD_ROOT/tmp"
export APPTAINER_TMPDIR="$BUILD_ROOT/tmp"
STAGE="$BUILD_ROOT/images"
mkdir -p "$TMPDIR" "$STAGE"

IMAGES_DIR="${T1_SOFTWARE_IMAGES_DIR:-$REPO_ROOT/images}"
mkdir -p "$IMAGES_DIR"

# To clear this storage, use `pm system reset -f`; plain `rm -rf` fails on
# busy overlay mounts.
pm() {
    podman --root "$BUILD_ROOT/podman/root" --runroot "$BUILD_ROOT/podman/run" \
        --storage-opt overlay.ignore_chown_errors=true "$@"
}

# podman's --build-arg-file wants strict KEY=VALUE lines; strip the
# comment/blank lines our versions.env files carry for humans.
argfile() {
    grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$1"
}

# Look up one pin's value for the manifest, from the same versions.env
# podman itself reads.
pin() {
    argfile "$1" | grep "^${2}=" | cut -d= -f2-
}

# Tag: <YYYYMMDD>-<short-git-sha>[-dirty]. Ties every image back to the
# exact Containerfile/versions.env content that built it, independent of
# the upstream pins recorded in the manifest below.
GIT_SHA="$(git rev-parse --short=8 HEAD 2>/dev/null || echo nogit)"
if ! git diff --quiet HEAD -- . 2>/dev/null; then
    GIT_SHA="${GIT_SHA}-dirty"
fi
TAG="$(date -u +%Y%m%d)-${GIT_SHA}"
BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Atomic symlink update: build the new link under a temp name, then
# rename over the old one, so a concurrent reader never sees a
# missing/half-written -latest link.
update_latest() {
    local target="$1" linkname="$2" tmp="${2}.tmp.$$"
    ln -s "$target" "$tmp"
    mv -Tf "$tmp" "$linkname"
}

# Sidecar JSON manifest: build metadata + this layer's upstream pins.
# write_manifest <path> key=value [key=value ...]
write_manifest() {
    local manifest="$1"; shift
    {
        echo "{"
        echo "  \"tag\": \"${TAG}\","
        echo "  \"cuda_variant\": \"${VARIANT}\","
        echo "  \"built_at\": \"${BUILT_AT}\","
        echo "  \"t1_software_commit\": \"$(git rev-parse HEAD 2>/dev/null || echo unknown)\","
        echo "  \"pins\": {"
        local n=$# i=0
        for kv in "$@"; do
            i=$((i + 1))
            local sep=","
            [[ $i -eq $n ]] && sep=""
            echo "    \"${kv%%=*}\": \"${kv#*=}\"${sep}"
        done
        echo "  }"
        echo "}"
    } > "$manifest"
}

BASE_REF="localhost/t1-base-${VARIANT}:${TAG}"
LARTPC_REF="localhost/t1-lartpc-${VARIANT}:${TAG}"
BASE_ARGFILE="$TMPDIR/base.args"
LARTPC_ARGFILE="$TMPDIR/lartpc.args"
argfile base/versions.env > "$BASE_ARGFILE"
argfile subgroups/lartpc/versions.env > "$LARTPC_ARGFILE"

# Pinned: a registry upgrade must never change the on-disk cache format
# silently.
REGISTRY_IMAGE=docker.io/library/registry:2.8.3
CACHE_ARGS=()
REGISTRY_NAME=""
stop_registry() {
    [[ -n "$REGISTRY_NAME" ]] && pm rm -f -t 0 "$REGISTRY_NAME" > /dev/null 2>&1 || true
    REGISTRY_NAME=""
}
trap stop_registry EXIT
if [[ -n "${T1_SOFTWARE_CACHE_DIR:-}" ]]; then
    mkdir -p "$T1_SOFTWARE_CACHE_DIR/registry"
    exec 9> "$T1_SOFTWARE_CACHE_DIR/.lock"
    if ! flock -n 9; then
        echo "nightly-build: another build holds $T1_SOFTWARE_CACHE_DIR/.lock; retry later or unset T1_SOFTWARE_CACHE_DIR" >&2
        exit 1
    fi
    REGISTRY_NAME="t1-cache-registry-$$"
    # Random free host port, loopback only: nothing off-node can reach it.
    pm run -d --name "$REGISTRY_NAME" -p 127.0.0.1::5000 \
        -v "$T1_SOFTWARE_CACHE_DIR/registry:/var/lib/registry" \
        "$REGISTRY_IMAGE" > /dev/null
    REGISTRY="127.0.0.1:$(pm port "$REGISTRY_NAME" 5000/tcp | cut -d: -f2)"
    for _ in $(seq 30); do
        curl -fs "http://$REGISTRY/v2/" > /dev/null && break
        sleep 1
    done
    curl -fs "http://$REGISTRY/v2/" > /dev/null || { echo "nightly-build: cache registry did not start" >&2; exit 1; }
    echo "== shared layer cache: $T1_SOFTWARE_CACHE_DIR (registry $REGISTRY) =="
    # Plain HTTP is acceptable: the registry listens on loopback only.
    CACHE_ARGS=(--tls-verify=false)
fi
cache_args() {  # cache_args <repo>
    [[ -n "$REGISTRY_NAME" ]] || return 0
    printf '%s\n' "${CACHE_ARGS[@]}" --cache-from "$REGISTRY/$1" --cache-to "$REGISTRY/$1"
}
mapfile -t BASE_CACHE < <(cache_args t1-base-cache)
mapfile -t LARTPC_CACHE < <(cache_args t1-lartpc-cache)

echo "== podman build ${BASE_REF} =="
pm build "${BASE_CACHE[@]}" --build-arg-file "$BASE_ARGFILE" \
    -f base/Containerfile -t "$BASE_REF" base

echo "== podman build ${LARTPC_REF} =="
pm build "${LARTPC_CACHE[@]}" --build-arg-file "$LARTPC_ARGFILE" \
    --build-arg "BASE_IMAGE=${BASE_REF}" \
    -f subgroups/lartpc/Containerfile -t "$LARTPC_REF" .

# The cache is needed only by the builds; release it for other builds now.
stop_registry
exec 9>&-

echo "== image tests under podman (CPU) =="
pm run --rm "$BASE_REF" bash /opt/t1-base-tests/image-test.sh
pm run --rm "$LARTPC_REF" bash /opt/t1-lartpc-tests/image-test.sh

BASE_TAR="$STAGE/t1-base-${VARIANT}-${TAG}.oci.tar"
LARTPC_TAR="$STAGE/t1-lartpc-${VARIANT}-${TAG}.oci.tar"
LARTPC_SIF="$STAGE/t1-lartpc-${VARIANT}-${TAG}.sif"
echo "== podman save -> OCI archives =="
rm -f "$BASE_TAR" "$LARTPC_TAR"
pm save --format oci-archive -o "$BASE_TAR" "$BASE_REF"
pm save --format oci-archive -o "$LARTPC_TAR" "$LARTPC_REF"

echo "== apptainer build $(basename "$LARTPC_SIF") from the OCI archive =="
# mksquashfs sizes its buffer at 25% of the node's physical RAM and
# ignores the job's cgroup limit, so in a small Slurm allocation the
# kernel OOM-kills it. A fixed 2G buffer fits any allocation that can
# run the tests at all.
apptainer build --force --mksquashfs-args "-mem 2G" "$LARTPC_SIF" "oci-archive://$LARTPC_TAR"

echo "== image tests under apptainer =="
GPU_FLAG=()
nvidia-smi -L > /dev/null 2>&1 && GPU_FLAG=(--nv)
apptainer exec "${GPU_FLAG[@]}" "$LARTPC_SIF" bash /opt/t1-base-tests/image-test.sh
apptainer exec "${GPU_FLAG[@]}" "$LARTPC_SIF" bash /opt/t1-lartpc-tests/image-test.sh

echo "== dprod full-chain test (edep-sim -> JAXTPC -> pysupera, GPU) =="
scripts/dprod-pipeline-test.sh "$LARTPC_SIF"

# Every build and test above passed -- only now move the finished files
# off local scratch into the (typically NFS) images dir, one big
# sequential transfer per file. set -e means a failed build or test never
# reaches here, so a broken image never lands in $IMAGES_DIR.
for f in "$BASE_TAR" "$LARTPC_TAR" "$LARTPC_SIF"; do
    mv "$f" "$IMAGES_DIR/"
done

update_latest "$(basename "$BASE_TAR")" "$IMAGES_DIR/t1-base-${VARIANT}-latest.oci.tar"
write_manifest "$IMAGES_DIR/t1-base-${VARIANT}-${TAG}.manifest.json" \
    "torch_version=$(pin base/versions.env TORCH_VERSION)" \
    "jax_version=$(pin base/versions.env JAX_VERSION)" \
    "root_version=$(pin base/versions.env ROOT_VERSION)" \
    "geant4_version=$(pin base/versions.env GEANT4_VERSION)"

update_latest "$(basename "$LARTPC_TAR")" "$IMAGES_DIR/t1-lartpc-${VARIANT}-latest.oci.tar"
update_latest "$(basename "$LARTPC_SIF")" "$IMAGES_DIR/t1-lartpc-${VARIANT}-latest.sif"
write_manifest "$IMAGES_DIR/t1-lartpc-${VARIANT}-${TAG}.manifest.json" \
    "jaxtpc_commit=$(pin subgroups/lartpc/versions.env JAXTPC_COMMIT)" \
    "pimm_data_commit=$(pin subgroups/lartpc/versions.env PIMM_DATA_COMMIT)" \
    "goop_commit=$(pin subgroups/lartpc/versions.env GOOP_COMMIT)" \
    "chroma_lar_commit=$(pin subgroups/lartpc/versions.env CHROMA_LAR_COMMIT)" \
    "trichroma_commit=$(pin subgroups/lartpc/versions.env TRICHROMA_COMMIT)" \
    "photonlib_commit=$(pin subgroups/lartpc/versions.env PHOTONLIB_COMMIT)" \
    "trichroma_triton_version=$(pin subgroups/lartpc/versions.env TRICHROMA_TRITON_VERSION)" \
    "dlpgenerator_commit=$(pin subgroups/lartpc/versions.env DLPGENERATOR_COMMIT)" \
    "edepsim_commit=$(pin subgroups/lartpc/versions.env EDEPSIM_COMMIT)" \
    "pysupera_commit=$(pin subgroups/lartpc/versions.env PYSUPERA_COMMIT)" \
    "dprod_commit=$(pin subgroups/lartpc/versions.env DPROD_COMMIT)"

echo "== done: $IMAGES_DIR/{$(basename "$BASE_TAR"),$(basename "$LARTPC_TAR"),$(basename "$LARTPC_SIF")} =="
echo "== latest: $IMAGES_DIR/t1-lartpc-${VARIANT}-latest.{sif,oci.tar} $IMAGES_DIR/t1-base-${VARIANT}-latest.oci.tar =="
