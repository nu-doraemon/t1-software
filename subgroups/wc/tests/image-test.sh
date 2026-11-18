#!/usr/bin/env bash
# t1-wc image checks. Runs inside the image under either runtime:
#   podman run --rm t1-wc-cuda12:<tag> bash /opt/t1-wc-tests/image-test.sh
#   apptainer exec [--nv] t1-wc-cuda12.sif bash /opt/t1-wc-tests/image-test.sh
# The base's own checks (/opt/t1-base-tests/image-test.sh) run on this
# image too, separately.
set -euo pipefail
[ "$(command -v python3)" = /opt/conda/bin/python3 ]

echo "== base packages unchanged; LUCiD imports =="
python3 - <<'EOF'
import importlib.metadata as m
for line in open("/opt/t1-wc/base-constraints.txt").read().split():
    name, pinned = line.split("==")
    assert m.version(name) == pinned, f"{name}: {m.version(name)} != base {pinned}"
import lucid, flax, optax
print("lucid", m.version("lucid-sim"), lucid.__file__)
print("flax", flax.__version__, "optax", optax.__version__)
EOF

echo "== base Geant4/ROOT unchanged; PhotonSim's own Geant4 =="
[ "$(command -v geant4-config)" = /opt/conda/bin/geant4-config ]
[ "$(printf 'geant4=%s\nroot=%s\n' "$(geant4-config --version)" "$(root-config --version)")" \
    = "$(cat /opt/t1-wc/base-g4root-versions)" ]
[ "$(/opt/photonsim-g4/bin/geant4-config --version)" = "$(cat /opt/t1-wc/photonsim-geant4-version)" ]
[ -x "$PHOTONSIM_BIN" ]

echo "== provenance LUCiD writes into every dataset batch =="
python3 - <<'EOF'
import os
from lucid.utils import code_version, photonsim_version
commit, dirty, ref = code_version()
ps = photonsim_version(os.environ["PHOTONSIM_BIN"])
print("lucid", commit, "dirty", dirty, "ref", ref)
print("photonsim", ps)
assert len(commit) == 40 and dirty is False, (commit, dirty)
assert len(ps["photonsim_commit"]) == 40, ps
assert "unknown" not in (ps["photonsim_ref"], ps["geant4_version"]), ps
EOF

echo "== PhotonSim -> LUCiD seam =="
bash /opt/t1-wc-tests/test_photonsim_lucid.sh

# LUCiD's default collection (its conftest forces JAX onto the CPU and
# leaves the --slow files out). Tests that need LUCiD's downloaded SIREN
# data, or a byte-identity input named by an env var, skip.
echo "== LUCiD unit suite =="
cd /opt/LUCiD
python3 -m pytest tests/ -q -p no:cacheprovider -rs
echo "t1-wc image-test: PASSED"
