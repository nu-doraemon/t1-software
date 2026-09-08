#!/usr/bin/env bash
# Nightly (or on-demand) build of the T1-LArTPC container stack.
#
#   scripts/nightly-build.sh [cuda12|cuda13]   # default: cuda12
#
# Apptainer's build scratch/cache is redirected to $LSCRATCH (large local
# disk -- building a multi-GB image under a small /tmp fails or thrashes).
# Final .sif images and their manifests land in $T1_SOFTWARE_IMAGES_DIR,
# default ./images next to this repo; set the env var to redirect them to
# a shared/persistent location instead (this repo never assumes one). No
# docker/podman daemon; unprivileged apptainer --fakeroot only (confirmed
# working without a /etc/subuid or /etc/subgid entry on the nodes this
# has been built on so far).
#
# Versioning: see README.md "Image versioning". Every image built here is
# tagged <layer>-<variant>-<date>-<git-sha>[-dirty].sif; a `-latest.sif`
# symlink per layer+variant always points at the newest one, and a JSON
# manifest sits next to each image recording the pins that produced it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${1:-cuda12}"
case "$VARIANT" in
    cuda12) ;;
    cuda13) echo "nightly-build: cuda13 not built by this script yet (no verified >=580.65.06 driver host to build/test it on)." >&2; exit 1 ;;
    *) echo "nightly-build: unknown variant '$VARIANT' (expected cuda12)" >&2; exit 1 ;;
esac

: "${LSCRATCH:?nightly-build: \$LSCRATCH is not set (needs a batch/interactive allocation that provides it)}"
export APPTAINER_TMPDIR="$LSCRATCH/t1-software-build/tmp"
export APPTAINER_CACHEDIR="$LSCRATCH/t1-software-build/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

IMAGES_DIR="${T1_SOFTWARE_IMAGES_DIR:-$REPO_ROOT/images}"
mkdir -p "$IMAGES_DIR"

# apptainer's --build-arg-file wants strict KEY=VALUE lines; strip the
# comment/blank lines our versions.env files carry for humans.
argfile() {
    grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$1"
}

# Look up one pin's value for the manifest, from the same versions.env
# apptainer itself reads.
pin() {
    argfile "$1" | grep "^${2}=" | cut -d= -f2-
}

# Tag: <YYYYMMDD>-<short-git-sha>[-dirty]. Ties every image back to the
# exact .def/versions.env content that built it, independent of the
# upstream (JAXTPC/pimm-data) pins recorded in the manifest below.
GIT_SHA="$(git rev-parse --short=8 HEAD 2>/dev/null || echo nogit)"
if ! git diff --quiet HEAD -- . 2>/dev/null; then
    GIT_SHA="${GIT_SHA}-dirty"
fi
TAG="$(date -u +%Y%m%d)-${GIT_SHA}"
BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Atomic symlink update: build the new link under a temp name, then
# rename over the old one, so a concurrent reader never sees a
# missing/half-written -latest.sif.
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

echo "== building t1-base-${VARIANT} (${TAG}) =="
BASE_ARGFILE="$(mktemp)"
LARTPC_ARGFILE="$(mktemp)"
trap 'rm -f "$BASE_ARGFILE" "$LARTPC_ARGFILE"' EXIT
argfile base/versions.env > "$BASE_ARGFILE"
BASE_IMAGE="$IMAGES_DIR/t1-base-${VARIANT}-${TAG}.sif"
# CUDA_VARIANT is documentation-only in base.def (the docker From: tag is
# fixed, not templated) -- downgrade apptainer's unused-build-arg check to
# a warning instead of failing the build on it.
apptainer build --fakeroot --force --warn-unused-build-args \
    --build-arg-file "$BASE_ARGFILE" \
    "$BASE_IMAGE" base/base.def
update_latest "$(basename "$BASE_IMAGE")" "$IMAGES_DIR/t1-base-${VARIANT}-latest.sif"
write_manifest "$IMAGES_DIR/t1-base-${VARIANT}-${TAG}.manifest.json" \
    "torch_version=$(pin base/versions.env TORCH_VERSION)" \
    "jax_version=$(pin base/versions.env JAX_VERSION)"

echo "== building t1-lartpc-${VARIANT} (${TAG}) =="
argfile subgroups/lartpc/versions.env > "$LARTPC_ARGFILE"
LARTPC_IMAGE="$IMAGES_DIR/t1-lartpc-${VARIANT}-${TAG}.sif"
apptainer build --fakeroot --force \
    --build-arg-file "$LARTPC_ARGFILE" \
    --build-arg "BASE_SIF=${BASE_IMAGE}" \
    "$LARTPC_IMAGE" subgroups/lartpc/lartpc.def
update_latest "$(basename "$LARTPC_IMAGE")" "$IMAGES_DIR/t1-lartpc-${VARIANT}-latest.sif"
write_manifest "$IMAGES_DIR/t1-lartpc-${VARIANT}-${TAG}.manifest.json" \
    "jaxtpc_commit=$(pin subgroups/lartpc/versions.env JAXTPC_COMMIT)" \
    "pimm_data_commit=$(pin subgroups/lartpc/versions.env PIMM_DATA_COMMIT)"

echo "== running lartpc %test (includes JAXTPC -> pimm-data integration test) =="
apptainer test "$LARTPC_IMAGE"

echo "== done: ${BASE_IMAGE} ${LARTPC_IMAGE} =="
echo "== latest: $IMAGES_DIR/t1-base-${VARIANT}-latest.sif $IMAGES_DIR/t1-lartpc-${VARIANT}-latest.sif =="
