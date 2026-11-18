#!/usr/bin/env bash
# Build the T1 container stack: the shared base image and every subgroup
# overlay (lartpc, wc) on top of it, their tests, and the apptainer .sifs.
#
#   scripts/build.sh [--variant cuda12] [--work-dir DIR] [--images-dir DIR]
#                    [--cache-dir DIR] [--pin KEY=VALUE]... [--subgroup NAME]...
#                    [--tag TAG] [--no-sif] [--no-base-archive] [--no-pipeline-test]
#                    [--gpu auto|on|off] [--squashfs-mem SIZE] [--help]
#
# `scripts/build.sh --help` lists each option with its default. Three
# options fall back to an environment variable when not given:
#   --images-dir        $T1_SOFTWARE_IMAGES_DIR (else ./images next to this repo)
#   --cache-dir         $T1_SOFTWARE_CACHE_DIR  (else node-local cache only)
#   --no-pipeline-test  on when T1_SKIP_PIPELINE_TEST=1
#
# Without --subgroup, every subgroup in SUBGROUPS_ALL is built; the base
# is built once and shared. Without --pin, the build uses the pins in
# base/versions.env and subgroups/<name>/versions.env. A release build is
# that default path with a version --tag. --pin KEY=VALUE overrides one
# pin for this build only; it never edits a file. Its key may come from
# any subgroup's versions.env, built this run or not, so one --pin set
# serves every --subgroup selection.
#
# Outputs, in the images dir:
#   t1-base-<variant>-<tag>.oci.tar     base only; `podman load` it to
#                                       build an overlay elsewhere (not
#                                       with --no-base-archive)
#   t1-<sub>-<variant>-<tag>.oci.tar    per subgroup; `podman load -i` / `docker load -i`
#   t1-<sub>-<variant>-<tag>.sif        apptainer image of the same OCI image
#   <layer>-<variant>-<tag>.manifest.json, and -latest symlinks per file kind
#
# Rootless podman without a /etc/subuid entry maps a single UID, so:
#   - overlay.ignore_chown_errors=true lets layers that chown to other
#     users unpack (every file ends up owned by the build user);
#   - the Containerfiles run apt's sandbox as root.
# Podman storage, temp files and staged outputs all live in the work dir.
# Put it on a node-local disk: rootless overlay needs a local filesystem,
# and writing multi-GB files straight to a shared/NFS images dir is much
# slower than one local write plus one move. The layer cache in that
# storage is reused by the next build on the same node only.
#
# Shared layer cache (optional): --cache-dir names a persistent directory
# on shared storage. For the builds, the script starts a registry
# container (REGISTRY_IMAGE, below) on 127.0.0.1 with its blob store in
# <cache-dir>/registry, and builds with --cache-from/--cache-to against
# it. A build on any node then reuses every step whose inputs are
# unchanged. One build at a time may use the cache: a second one fails
# fast on <cache-dir>/.lock. The cache is disposable -- delete the
# directory to reclaim space; the next build repopulates it.
#
# Tests: the base's and each subgroup's image-test.sh run under podman
# (CPU) and under apptainer (--nv when a GPU is present); both see the
# image's static ENV with no setup step. When lartpc is built,
# dprod-pipeline-test.sh then runs on its .sif. Nothing leaves the work
# dir unless all of them pass.
#
# Versioning: see README.md "Image versioning". The default tag is
# <date>-<git-sha>[-dirty]; a JSON manifest next to each layer records
# every pin that produced it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Every subgroup overlay, in build order. A new subgroup adds its name
# here (and a subgroups/<name>/ directory with Containerfile,
# versions.env and tests/image-test.sh).
SUBGROUPS_ALL=(lartpc wc)

die_usage() {
    echo "build.sh: $*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
usage: scripts/build.sh [options]

  --variant NAME         CUDA variant to build (default: cuda12; only cuda12 is built)
  --work-dir DIR         podman storage, temp files and staged outputs
                         (default: $LSCRATCH/t1-software-build if $LSCRATCH is set,
                         else ${TMPDIR:-/tmp}/t1-software-build)
  --images-dir DIR       where finished images go
                         (default: $T1_SOFTWARE_IMAGES_DIR, else <repo>/images)
  --cache-dir DIR        shared layer cache (default: $T1_SOFTWARE_CACHE_DIR;
                         unset means node-local cache only)
  --pin KEY=VALUE        override one versions.env key for this build; repeatable;
                         a key in no versions.env file, or in more than one, is an error
  --subgroup NAME        build only this subgroup overlay; repeatable
                         (default: all of lartpc wc)
  --tag TAG              image tag (default: <YYYYMMDD>-<git-sha8>[-dirty])
  --no-sif               skip the .sifs, the apptainer tests and the pipeline test
  --no-base-archive      do not save the t1-base .oci.tar or its -latest link; the
                         base is still built, tested and recorded in its manifest
  --no-pipeline-test     skip the dprod full-chain test (default: off;
                         on when T1_SKIP_PIPELINE_TEST=1)
  --gpu auto|on|off      apptainer --nv: auto probes nvidia-smi (default: auto)
  --squashfs-mem SIZE    mksquashfs buffer for the .sif (default: 2G)
  --help                 show this help
EOF
}

VARIANT=cuda12
if [[ -n "${LSCRATCH:-}" ]]; then
    WORK_DIR="$LSCRATCH/t1-software-build"
else
    WORK_DIR="${TMPDIR:-/tmp}/t1-software-build"
fi
IMAGES_DIR="${T1_SOFTWARE_IMAGES_DIR:-$REPO_ROOT/images}"
CACHE_DIR="${T1_SOFTWARE_CACHE_DIR:-}"
PINS=()
REQUESTED=()
TAG=""
BUILD_SIF=1
BASE_ARCHIVE=1
PIPELINE_TEST=1
[[ "${T1_SKIP_PIPELINE_TEST:-0}" == "1" ]] && PIPELINE_TEST=0
GPU=auto
SQUASHFS_MEM=2G

# need_value <option> <argc>: a value option needs one more argument.
need_value() {
    (( $2 >= 2 )) || die_usage "$1 needs a value"
}

while (($#)); do
    opt="$1"
    val=""
    case "$opt" in
        --help=*|--no-sif=*|--no-base-archive=*|--no-pipeline-test=*) die_usage "${opt%%=*} takes no value" ;;
        --*=*) val="${opt#*=}"; opt="${opt%%=*}"; set -- "$opt" "$val" "${@:2}" ;;
    esac
    case "$opt" in
        --help) usage; exit 0 ;;
        --no-sif) BUILD_SIF=0; shift ;;
        --no-base-archive) BASE_ARCHIVE=0; shift ;;
        --no-pipeline-test) PIPELINE_TEST=0; shift ;;
        --variant|--work-dir|--images-dir|--cache-dir|--pin|--subgroup|--tag|--gpu|--squashfs-mem)
            need_value "$opt" "$#"
            val="$2"
            case "$opt" in
                --variant) VARIANT="$val" ;;
                --work-dir) WORK_DIR="$val" ;;
                --images-dir) IMAGES_DIR="$val" ;;
                --cache-dir) CACHE_DIR="$val" ;;
                --pin) PINS+=("$val") ;;
                --subgroup) REQUESTED+=("$val") ;;
                --tag) TAG="$val" ;;
                --gpu) GPU="$val" ;;
                --squashfs-mem) SQUASHFS_MEM="$val" ;;
            esac
            shift 2 ;;
        *) die_usage "unknown option '$opt' (see --help)" ;;
    esac
done

case "$VARIANT" in
    cuda12) ;;
    cuda13) echo "build.sh: cuda13 not built by this script yet (no verified >=580.65.06 driver host to build/test it on)." >&2; exit 1 ;;
    *) die_usage "unknown variant '$VARIANT' (expected cuda12)" ;;
esac
case "$GPU" in
    auto|on|off) ;;
    *) die_usage "--gpu must be auto, on or off (got '$GPU')" ;;
esac
[[ -n "$WORK_DIR" && -n "$IMAGES_DIR" ]] || die_usage "--work-dir and --images-dir must not be empty"

for sub in "${REQUESTED[@]}"; do
    case " ${SUBGROUPS_ALL[*]} " in
        *" $sub "*) ;;
        *) die_usage "unknown subgroup '$sub' (expected: ${SUBGROUPS_ALL[*]})" ;;
    esac
done
# SELECTED keeps SUBGROUPS_ALL's order, whatever order --subgroup came in.
SELECTED=()
for sub in "${SUBGROUPS_ALL[@]}"; do
    if (( ${#REQUESTED[@]} == 0 )) || [[ " ${REQUESTED[*]} " == *" $sub "* ]]; then
        SELECTED+=("$sub")
    fi
done
is_selected() {
    [[ " ${SELECTED[*]} " == *" $1 "* ]]
}

# Absolute before the cd below: a relative path is meant from the
# caller's cwd, and podman's -v reads a relative source as a volume name.
WORK_DIR="$(realpath -m "$WORK_DIR")"
IMAGES_DIR="$(realpath -m "$IMAGES_DIR")"
[[ -n "$CACHE_DIR" ]] && CACHE_DIR="$(realpath -m "$CACHE_DIR")"
cd "$REPO_ROOT"

export TMPDIR="$WORK_DIR/tmp"
export APPTAINER_TMPDIR="$WORK_DIR/tmp"
STAGE="$WORK_DIR/images"
mkdir -p "$TMPDIR" "$STAGE"

# podman's --build-arg-file wants strict KEY=VALUE lines; strip the
# comment/blank lines our versions.env files carry for humans.
argfile() {
    grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$1"
}

# One effective argument file per layer, for every subgroup (selected or
# not), so --pin can resolve any key. ARGFILES[base] first, then the
# subgroups in SUBGROUPS_ALL order.
declare -A ARGFILES
ARGFILES[base]="$TMPDIR/base.args"
argfile base/versions.env > "${ARGFILES[base]}"
for sub in "${SUBGROUPS_ALL[@]}"; do
    ARGFILES[$sub]="$TMPDIR/$sub.args"
    argfile "subgroups/$sub/versions.env" > "${ARGFILES[$sub]}"
done
LAYERS=(base "${SUBGROUPS_ALL[@]}")

# has_key <argfile> <key>
has_key() {
    grep -q "^${2}=" "$1"
}

# set_key <argfile> <key> <value>: replace the key's line in place. awk
# reads key and value from ENVIRON, which (unlike -v) applies no escape
# processing, so any value passes through verbatim.
set_key() {
    local tmp="$1.tmp"
    PIN_KEY="$2" PIN_VALUE="$3" awk '
        BEGIN { k = ENVIRON["PIN_KEY"]; v = ENVIRON["PIN_VALUE"] }
        index($0, k "=") == 1 { print k "=" v; next }
        { print }
    ' "$1" > "$tmp"
    mv "$tmp" "$1"
}

for kv in "${PINS[@]}"; do
    [[ "$kv" == *=* ]] || die_usage "--pin needs KEY=VALUE (got '$kv')"
    key="${kv%%=*}"
    value="${kv#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die_usage "--pin key '$key' is not a valid name"
    owners=()
    for layer in "${LAYERS[@]}"; do
        if has_key "${ARGFILES[$layer]}" "$key"; then owners+=("$layer"); fi
    done
    case "${#owners[@]}" in
        0) die_usage "--pin $key is not in base/versions.env or any subgroups/*/versions.env" ;;
        1) set_key "${ARGFILES[${owners[0]}]}" "$key" "$value" ;;
        *) die_usage "--pin $key is in more than one versions.env file (${owners[*]})" ;;
    esac
done

# An empty <PKG>_REPO would reach `git clone ""` inside the build and fail
# there with an unclear error; catch it here.
for layer in "${LAYERS[@]}"; do
    while IFS= read -r line; do
        key="${line%%=*}"
        if [[ "$key" == *_REPO && -z "${line#*=}" ]]; then
            die_usage "$key is empty"
        fi
    done < "${ARGFILES[$layer]}"
done

mkdir -p "$IMAGES_DIR"

# To clear this storage, use `pm system reset -f`; plain `rm -rf` fails on
# busy overlay mounts.
pm() {
    podman --root "$WORK_DIR/podman/root" --runroot "$WORK_DIR/podman/run" \
        --storage-opt overlay.ignore_chown_errors=true "$@"
}

# Default tag: <YYYYMMDD>-<short-git-sha>[-dirty]. Ties every image back
# to the exact Containerfile/versions.env content that built it,
# independent of the upstream pins recorded in the manifest below.
if [[ -z "$TAG" ]]; then
    GIT_SHA="$(git rev-parse --short=8 HEAD 2>/dev/null || echo nogit)"
    if ! git diff --quiet HEAD -- . 2>/dev/null; then
        GIT_SHA="${GIT_SHA}-dirty"
    fi
    TAG="$(date -u +%Y%m%d)-${GIT_SHA}"
fi
BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
T1_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

# Atomic symlink update: build the new link under a temp name, then
# rename over the old one, so a concurrent reader never sees a
# missing/half-written -latest link.
update_latest() {
    local target="$1" linkname="$2" tmp="${2}.tmp.$$"
    ln -s "$target" "$tmp"
    mv -Tf "$tmp" "$linkname"
}

json_escape() {
    local s="${1//\\/\\\\}"
    printf '%s' "${s//\"/\\\"}"
}

# Sidecar JSON manifest: build metadata + every pin of the layer, taken
# from the effective argument file podman read (so --pin overrides show).
# write_manifest <path> <argfile>
write_manifest() {
    local manifest="$1" args="$2"
    local n i=0 line key
    n="$(wc -l < "$args")"
    {
        echo "{"
        echo "  \"tag\": \"$(json_escape "$TAG")\","
        echo "  \"cuda_variant\": \"${VARIANT}\","
        echo "  \"built_at\": \"${BUILT_AT}\","
        echo "  \"t1_software_commit\": \"${T1_COMMIT}\","
        echo "  \"pins\": {"
        while IFS= read -r line; do
            i=$((i + 1))
            local sep=","
            [[ $i -eq $n ]] && sep=""
            key="${line%%=*}"
            echo "    \"${key,,}\": \"$(json_escape "${line#*=}")\"${sep}"
        done < "$args"
        echo "  }"
        echo "}"
    } > "$manifest"
}

# ref <layer>: the local image reference of a layer.
ref() {
    echo "localhost/t1-$1-${VARIANT}:${TAG}"
}
BASE_REF="$(ref base)"

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
if [[ -n "$CACHE_DIR" ]]; then
    mkdir -p "$CACHE_DIR/registry"
    exec 9> "$CACHE_DIR/.lock"
    if ! flock -n 9; then
        echo "build.sh: another build holds $CACHE_DIR/.lock; retry later or drop --cache-dir" >&2
        exit 1
    fi
    REGISTRY_NAME="t1-cache-registry-$$"
    # Random free host port, loopback only: nothing off-node can reach it.
    pm run -d --name "$REGISTRY_NAME" -p 127.0.0.1::5000 \
        -v "$CACHE_DIR/registry:/var/lib/registry" \
        "$REGISTRY_IMAGE" > /dev/null
    REGISTRY="127.0.0.1:$(pm port "$REGISTRY_NAME" 5000/tcp | cut -d: -f2)"
    for _ in $(seq 30); do
        curl -fs "http://$REGISTRY/v2/" > /dev/null && break
        sleep 1
    done
    curl -fs "http://$REGISTRY/v2/" > /dev/null || { echo "build.sh: cache registry did not start" >&2; exit 1; }
    echo "== shared layer cache: $CACHE_DIR (registry $REGISTRY) =="
    # Plain HTTP is acceptable: the registry listens on loopback only.
    CACHE_ARGS=(--tls-verify=false)
fi
cache_args() {  # cache_args <repo>
    [[ -n "$REGISTRY_NAME" ]] || return 0
    printf '%s\n' "${CACHE_ARGS[@]}" --cache-from "$REGISTRY/$1" --cache-to "$REGISTRY/$1"
}

echo "== podman build ${BASE_REF} =="
mapfile -t layer_cache < <(cache_args t1-base-cache)
pm build "${layer_cache[@]}" --build-arg-file "${ARGFILES[base]}" \
    -f base/Containerfile -t "$BASE_REF" base

for sub in "${SELECTED[@]}"; do
    echo "== podman build $(ref "$sub") =="
    mapfile -t layer_cache < <(cache_args "t1-$sub-cache")
    pm build "${layer_cache[@]}" --build-arg-file "${ARGFILES[$sub]}" \
        --build-arg "BASE_IMAGE=${BASE_REF}" \
        -f "subgroups/$sub/Containerfile" -t "$(ref "$sub")" .
done

# The cache is needed only by the builds; release it for other builds now.
stop_registry
exec 9>&-

echo "== image tests under podman (CPU) =="
pm run --rm "$BASE_REF" bash /opt/t1-base-tests/image-test.sh
for sub in "${SELECTED[@]}"; do
    pm run --rm "$(ref "$sub")" bash "/opt/t1-$sub-tests/image-test.sh"
done

# staged <layer> <suffix>: the layer's file in the staging dir.
staged() {
    echo "$STAGE/t1-$1-${VARIANT}-${TAG}.$2"
}
echo "== podman save -> OCI archives =="
OUTPUTS=()
if (( BASE_ARCHIVE )); then
    rm -f "$(staged base oci.tar)"
    pm save --format oci-archive -o "$(staged base oci.tar)" "$BASE_REF"
    OUTPUTS+=("$(staged base oci.tar)")
fi
for sub in "${SELECTED[@]}"; do
    rm -f "$(staged "$sub" oci.tar)"
    pm save --format oci-archive -o "$(staged "$sub" oci.tar)" "$(ref "$sub")"
    OUTPUTS+=("$(staged "$sub" oci.tar)")
done

if (( BUILD_SIF )); then
    GPU_FLAG=()
    case "$GPU" in
        on) GPU_FLAG=(--nv) ;;
        auto) nvidia-smi -L > /dev/null 2>&1 && GPU_FLAG=(--nv) ;;
        off) ;;
    esac
    for sub in "${SELECTED[@]}"; do
        sif="$(staged "$sub" sif)"
        echo "== apptainer build $(basename "$sif") from the OCI archive =="
        # mksquashfs sizes its buffer at 25% of the node's physical RAM and
        # ignores the job's cgroup limit, so in a small Slurm allocation the
        # kernel OOM-kills it. A fixed buffer (--squashfs-mem, default 2G)
        # fits any allocation that can run the tests at all.
        apptainer build --force --mksquashfs-args "-mem ${SQUASHFS_MEM}" "$sif" \
            "oci-archive://$(staged "$sub" oci.tar)"
        OUTPUTS+=("$sif")

        echo "== image tests under apptainer: $sub =="
        apptainer exec "${GPU_FLAG[@]}" "$sif" bash /opt/t1-base-tests/image-test.sh
        apptainer exec "${GPU_FLAG[@]}" "$sif" bash "/opt/t1-$sub-tests/image-test.sh"
    done

    if ! is_selected lartpc; then
        echo "== dprod full-chain test: skipped (lartpc not selected) =="
    elif (( PIPELINE_TEST )); then
        echo "== dprod full-chain test (edep-sim -> JAXTPC -> pysupera, GPU) =="
        scripts/dprod-pipeline-test.sh --work-dir "$WORK_DIR" "$(staged lartpc sif)"
    else
        echo "== dprod full-chain test: skipped =="
    fi
fi

# Every build and test above passed -- only now move the finished files
# off local scratch into the (typically NFS) images dir, one big
# sequential transfer per file. set -e means a failed build or test never
# reaches here, so a broken image never lands in the images dir.
for f in "${OUTPUTS[@]}"; do
    mv "$f" "$IMAGES_DIR/"
done

# The base manifest is written even without the archive: it records the
# base pins under the overlay images.
if (( BASE_ARCHIVE )); then
    update_latest "t1-base-${VARIANT}-${TAG}.oci.tar" "$IMAGES_DIR/t1-base-${VARIANT}-latest.oci.tar"
fi
write_manifest "$IMAGES_DIR/t1-base-${VARIANT}-${TAG}.manifest.json" "${ARGFILES[base]}"

for sub in "${SELECTED[@]}"; do
    update_latest "t1-$sub-${VARIANT}-${TAG}.oci.tar" "$IMAGES_DIR/t1-$sub-${VARIANT}-latest.oci.tar"
    if (( BUILD_SIF )); then
        update_latest "t1-$sub-${VARIANT}-${TAG}.sif" "$IMAGES_DIR/t1-$sub-${VARIANT}-latest.sif"
    fi
    write_manifest "$IMAGES_DIR/t1-$sub-${VARIANT}-${TAG}.manifest.json" "${ARGFILES[$sub]}"
done

DONE_FILES=()
for f in "${OUTPUTS[@]}"; do DONE_FILES+=("$(basename "$f")"); done
echo "== done: $IMAGES_DIR: ${DONE_FILES[*]} =="
