#!/usr/bin/env bash
# PhotonSim -> LUCiD -> HDF5 seam: LUCiD's own production driver runs
# PhotonSim (through $PHOTONSIM_BIN) on a three-muon smoke config,
# converts its ROOT output and writes the four per-event HDF5 files
# (sensor, hits, step, labl), then verifies them itself. This test checks
# that verification and the files.
#   bash /opt/t1-wc-tests/test_photonsim_lucid.sh
set -euo pipefail
# apptainer passes the host's TMPDIR through; JAX's ptxas and mktemp
# fail when that directory does not exist inside the container.
[ -d "${TMPDIR:-/tmp}" ] || export TMPDIR=/tmp
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
cd "$out"

# --job-id and --output-dir are required by lucid-run-job; --master-seed
# seeds LUCiD's own sampling. This is a smoke test, not a
# reproducibility check.
lucid-run-job --config /opt/LUCiD/lucid/production/dataprod/configs/GeV/02_mu_smoke.json \
    --n-events 3 --job-id 1 --master-seed 42 --output-dir "$out/data" 2>&1 | tee "$out/run.log"

fail() { echo "test_photonsim_lucid: FAILED: $*" >&2; exit 1; }
for kind in sensor hits step labl; do
    n="$(find "$out/data/$kind" -maxdepth 1 -name '*.h5' | wc -l)"
    [ "$n" -eq 1 ] || fail "$kind: expected one .h5, found $n"
    grep -Eq "^ +OK ${kind} +size= *[0-9]+ .*n_events=3 complete=True$" "$out/run.log" \
        || fail "$kind: no 'OK $kind ... n_events=3 complete=True' verification line"
done
grep -Eq '^ +OK invariants ' "$out/run.log" || fail "no 'OK invariants' verification line"
if grep -q 'G4Exception' "$out/run.log"; then fail "PhotonSim raised a G4Exception"; fi
# The sidecar's Geant4 data must reach PhotonSim cleanly, not by
# overriding the base's values from a sourced hook.
if grep -q 'has been changed from' "$out/run.log"; then fail "Geant4 data variables were overridden"; fi
echo "test_photonsim_lucid: PASSED"
