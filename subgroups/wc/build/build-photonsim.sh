#!/bin/bash
# Build PhotonSim against the sidecar prefix /opt/photonsim-g4 (Geant4
# 11.3 + ROOT + GCC from conda-forge), run inside the t1-wc image build:
#   PHOTONSIM_REPO=... PHOTONSIM_COMMIT=... bash build-photonsim.sh
#
# Nothing of the base prefix (/opt/conda: Geant4 10.7, its ROOT, its
# compilers) may end up in the binary: two prefixes' zlib/xerces/
# libstdc++ in one process is not safe. So the build environment names
# only the sidecar, and the result is checked with readelf and ldd.
set -euo pipefail
: "${PHOTONSIM_REPO:?}" "${PHOTONSIM_COMMIT:?}"
SIDECAR=/opt/photonsim-g4

git clone --quiet "$PHOTONSIM_REPO" /opt/PhotonSim
git -C /opt/PhotonSim checkout --quiet "$PHOTONSIM_COMMIT"

(
    export PATH="$SIDECAR/bin:/usr/bin:/bin"
    export ROOTSYS="$SIDECAR" CONDA_PREFIX="$SIDECAR"
    export CONDA_BUILD_SYSROOT="$SIDECAR/x86_64-conda-linux-gnu/sysroot"
    unset LD_LIBRARY_PATH CMAKE_PREFIX_PATH
    # Only the compiler hooks: they set CC/CXX/CFLAGS/LDFLAGS for the
    # sidecar's own GCC. Third-party hooks are not nounset-safe.
    set +u
    for f in "$SIDECAR"/etc/conda/activate.d/*.sh; do
        case "$f" in *gcc_linux*|*gxx_linux*|*binutils_linux*) . "$f" ;; esac
    done
    set -u
    mkdir -p /opt/PhotonSim/build
    cd /opt/PhotonSim/build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$SIDECAR" ..
    make -j"$(nproc)"
)

bin=/opt/PhotonSim/build/PhotonSim
rpath="$(readelf -d "$bin" | grep -E 'R(UN)?PATH' || true)"
echo "build-photonsim: $rpath"
case "$rpath" in *"$SIDECAR/lib"*) ;; *) echo "build-photonsim: RPATH lacks $SIDECAR/lib" >&2; exit 1 ;; esac
case "$rpath" in *"/opt/conda"*) echo "build-photonsim: RPATH names /opt/conda" >&2; exit 1 ;; esac
libs="$(ldd "$bin")"
if grep -q 'not found' <<< "$libs"; then echo "$libs" >&2; echo "build-photonsim: unresolved libraries" >&2; exit 1; fi
if grep -q '=> /opt/conda/' <<< "$libs"; then echo "$libs" >&2; echo "build-photonsim: links the base prefix" >&2; exit 1; fi

# Version-free links to the sidecar's Geant4 datasets, from its own
# geant4-data hooks run in a clean environment (the hooks keep a value
# that is already set, so the image ENV must not leak in).
mkdir -p "$SIDECAR/g4data"
env -i CONDA_PREFIX="$SIDECAR" PATH=/usr/bin:/bin /bin/sh -c '
    for f in "$CONDA_PREFIX"/etc/conda/activate.d/*geant4-data*.sh; do . "$f" > /dev/null; done
    env' | grep -E '^G4[A-Z]*DATA=' | while IFS='=' read -r var value; do
    [ -d "$value" ] || { echo "build-photonsim: $var=$value is not a directory" >&2; exit 1; }
    ln -sfn "$value" "$SIDECAR/g4data/$var"
    echo "build-photonsim: $SIDECAR/g4data/$var -> $value"
done
[ -n "$(ls "$SIDECAR/g4data")" ] || { echo "build-photonsim: no Geant4 dataset found" >&2; exit 1; }

install -D -m 755 /tmp/photonsim-wrapper /opt/PhotonSim/bin/photonsim
