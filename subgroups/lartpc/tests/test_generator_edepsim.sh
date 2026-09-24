#!/usr/bin/env bash
# T1-LArTPC container smoke test: proves the DLPGenerator -> edep-sim seam
# that the "Event generation" ROADMAP.md item depends on -- a DLPGenerator
# YAML config, exported as an edep-sim "bomb" macro
# (`dlpgen ... --format bomb-macro`), must drive edep-sim's own
# "/generator/kinematics/set bomb" generator (drinkingkazu/edep-sim's
# src/kinem/ParticleBombGenerator, which links DLPGenerator's ParticleBomb
# directly) to produce a non-empty H5DLP output file.
#
# Runs entirely on CPU with a single low-multiplicity interaction (one
# muon) fired into edep-sim's own inputs/example.gdml -- no GPU, no
# network, deterministic (fixed SEED). Uses only the image's default
# environment (no setup script): generation via dlpgen + edep-sim, then
# the HDF5 output read back with h5py, proving the seam works for a
# plain `apptainer exec` user.
#
# Grounded against DLPGenerator commit 7b13a2a88d4f7a214ac84b53a66640392a50aec7's
# own python/dlp_generator/cli.py::_write_bomb_macro and
# edep-sim commit de64d94d6c6d3b6fa8eac42318064a99983faf81's
# app/edepSim.cc (persistency-manager-by-extension dispatch) and
# io/H5DLPFileStorage.h.
set -euo pipefail

# `-p /tmp`, not a bare `mktemp -d`: apptainer passes the host's TMPDIR
# through into the container by default (e.g. a batch scheduler that
# points it at a host scratch disk), and that path is not guaranteed to
# be bind-mounted inside the container -- `/tmp` always is.
WORKDIR="$(mktemp -d -p /tmp)"
trap 'rm -rf "$WORKDIR"' EXIT

cat > "$WORKDIR/bomb.yaml" <<'YAML'
SEED: 20260909
Interaction0:
  NumEvent: [1, 1]
  NumParticle: [1, 1]
  XRange: [0.0, 0.0]
  YRange: [0.0, 0.0]
  ZRange: [-250.0, -250.0]
  TRange: [0.0, 0.0]
  AddParent: false
  Particles:
    - PDG: [13]
      NumRange: [1, 1]
      KERange: [0.3, 0.3]
      Weight: 1.0
YAML

OUT_H5="$WORKDIR/out.h5"

(
    cd "$WORKDIR"

    dlpgen bomb.yaml 3 --format bomb-macro --output bomb.mac

    EXAMPLE_GDML=/opt/edep-sim-install/share/EDepSim/inputs/example.gdml
    if [ ! -e "$EXAMPLE_GDML" ]; then
        echo "example.gdml not found at $EXAMPLE_GDML" >&2
        exit 1
    fi

    # -C: this toy geometry has known overlaps upstream's own muon-100.mac
    # usage example disables the same way. -u: run /edep/update before the
    # macro (DLPGenerator's bomb-macro has no /edep/update of its own --
    # required once, before /run/beamOn, to initialize the run).
    edep-sim -C -u -g "$EXAMPLE_GDML" -o "$OUT_H5" bomb.mac
)

python3 - "$OUT_H5" <<'PY'
import sys

import h5py

path = sys.argv[1]

# Don't assume H5DLP's exact group layout (io/H5DLPDatatypes.h defines the
# per-record compound types -- Particle, Vertex, Primary, PStep -- but the
# group nesting itself is an implementation detail of
# EDepSimPersistencyManager.cc, not a stable contract this test should
# pin). Walk every dataset instead and require at least one non-empty row
# somewhere, which is exactly what "the seam produced real events" means.
datasets = {}
with h5py.File(path, "r") as f:
    f.visititems(
        lambda name, node: datasets.update({name: node.shape})
        if isinstance(node, h5py.Dataset)
        else None
    )

    if not datasets:
        raise SystemExit(f"H5DLP file has no datasets at all: {path}")

    nonempty = {name: shape for name, shape in datasets.items() if shape and shape[0] > 0}
    if not nonempty:
        raise SystemExit(
            f"H5DLP file has {len(datasets)} dataset(s) but every one is "
            f"empty (DLPGenerator -> edep-sim produced zero primaries?): "
            f"{sorted(datasets)}"
        )

    print(
        f"OK: DLPGenerator -> edep-sim seam produced {path} with "
        f"{len(nonempty)}/{len(datasets)} non-empty dataset(s): "
        f"{sorted(nonempty)}"
    )
PY
