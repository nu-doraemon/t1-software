#!/usr/bin/env bash
# Build-toolchain activation for compiling against ROOT/GEANT4 in this
# image (e.g. DLPGenerator, edep-sim). The runtime environment (python3,
# PyROOT, root, geant4 data) is already active in every container
# session; this adds only conda's compiler settings (CC/CXX/CFLAGS/
# LDFLAGS/CMAKE_ARGS for the GCC ROOT itself was built with), kept out of
# the default environment so unrelated builds never pick them up.
#
#   source /opt/setup-g4root.sh
#
export G4ROOT_CONDA_PREFIX=/opt/conda
export CONDA_PREFIX=/opt/conda
# Third-party hooks are not nounset-safe; relax -u just for them.
case "$-" in *u*) _g4root_had_nounset=1 ;; *) _g4root_had_nounset=0 ;; esac
set +u
for f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do
    case "$f" in *gcc_linux*|*gxx_linux*|*binutils_linux*) . "$f" ;; esac
done
unset f
[ "$_g4root_had_nounset" = 1 ] && set -u
unset _g4root_had_nounset
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:${CMAKE_PREFIX_PATH:-}"
