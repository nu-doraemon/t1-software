#!/usr/bin/env bash
# t1-base image checks. Runs inside the image under either runtime:
#   podman run --rm t1-base-cuda12:<tag> bash /opt/t1-base-tests/image-test.sh
#   apptainer exec t1-base-cuda12.sif bash /opt/t1-base-tests/image-test.sh
set -e
# Bare python3 must be the one interpreter carrying every stack.
[ "$(command -v python3)" = /opt/conda/bin/python3 ]
[ "$(command -v python)" = /opt/conda/bin/python ]
# torch imported BEFORE ROOT on purpose: the libstdc++ ordering case.
python3 -c "import torch, jax, ROOT; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('jax', jax.__version__, 'devices', jax.devices()); \
print('PyROOT', ROOT.gROOT.GetVersion()); \
ROOT.gInterpreter.Declare('#include <cassert>\\nint t1_probe(){assert(1); return 7;}'); \
assert ROOT.t1_probe() == 7"
root-config --version
geant4-config --version
[ -n "$G4NEUTRONHPDATA" ] && [ -d "$G4NEUTRONHPDATA" ]
# Compiler settings must stay out of the default environment.
[ -z "${CXXFLAGS:-}" ]
# No libstdc++ ABI-mismatch warning from cling (GXX_VERSION pin).
out="$(bash -c 'source /opt/setup-g4root.sh && root -b -q -l -e "int t1_probe = 0;" 2>&1')"
echo "$out"
# `! cmd` never trips set -e, so fail explicitly.
if echo "$out" | grep -q "standard library mismatch"; then exit 1; fi
echo "t1-base image-test: PASSED"
