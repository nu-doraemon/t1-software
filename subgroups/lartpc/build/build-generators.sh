#!/usr/bin/env bash
# Event generation, run inside the image by subgroups/lartpc/Containerfile:
# DLPGenerator (ParticleBomb) is built first, then edep-sim
# (drinkingkazu's fork -- adds src/kinem/ParticleBombGenerator, which
# links DLPGenerator's ParticleBomb directly into edep-sim's own
# "/generator/kinematics/set bomb" seam, so the build order is
# load-bearing). Both build with conda's own compiler via
# /opt/setup-g4root.sh -- the same GCC series ROOT/GEANT4 were built
# with. Pins come from the DLPGENERATOR_COMMIT and EDEPSIM_COMMIT build
# args. bash, not sh: DLPGenerator's setup.sh is bash-only
# (`${BASH_SOURCE[0]}`, `[[ ]]`). No `-u`: neither setup.sh nor conda's
# hooks are nounset-safe.
set -e
source /opt/setup-g4root.sh

cd /opt
git clone --quiet https://github.com/DeepLearnPhysics/DLPGenerator.git
cd DLPGenerator
git checkout --quiet "${DLPGENERATOR_COMMIT:?}"
# Upstream's setup.sh exports DLPGENERATOR_{DIR,LIBDIR,INCDIR,...};
# edep-sim's cmake below reads DLPGENERATOR_INCDIR/LIBDIR from the
# environment, so it stays sourced through edep-sim's build.
source setup.sh
make
# Upstream pytest suite through PyROOT: proves the ParticleBomb
# dictionary loads and its multiplicity/weight logic holds.
make test
chmod +x bin/dlpgen

cd /opt
git clone --quiet https://github.com/drinkingkazu/edep-sim.git
cd edep-sim
git checkout --quiet "${EDEPSIM_COMMIT:?}"
mkdir -p build && cd build
# EDEPSIM_DISPLAY=OFF: skips the interactive ROOT/G4 event-display GUI.
# RPATH bakes edep-sim's own lib dir, /opt/conda/lib, and DLPGenerator's
# build/lib into the installed binary. yaml-cpp comes from conda
# (above), whose include dir is already on conda's -isystem CXXFLAGS.
cmake $CMAKE_ARGS \
    -DCMAKE_INSTALL_PREFIX=/opt/edep-sim-install \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DEDEPSIM_DISPLAY=OFF \
    -DCMAKE_INSTALL_RPATH="/opt/edep-sim-install/lib;$CONDA_PREFIX/lib;$DLPGENERATOR_LIBDIR" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
    ..
make -j"$(nproc)"
make install
