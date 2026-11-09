#!/usr/bin/env bash
# Full production-chain test of a built t1-lartpc image.
#
#   scripts/dprod-pipeline-test.sh <t1-lartpc.sif> [events]     # default 5 events
#
# Runs dprod's own `dprod-install-test` inside the image: one small job
# through edep-sim -> JAXTPC (wire) -> pysupera -> JAXTPC (pixel) ->
# pysupera, with dprod's local scheduler (no slurm) and the in-image
# checkouts named by the site config baked into the image
# (/opt/t1-lartpc-tests/dprod_site_image.yaml). It exits non-zero unless
# every stage finishes and dprod's file checks (event ids, provenance
# chain) pass.
#
# Needs a GPU node: the JAXTPC stages run on the GPU (`--nv`). This cannot
# run during the image build, because the build has no GPU. Set
# T1_SKIP_PIPELINE_TEST=1 to skip it explicitly on a GPU-less node.
set -euo pipefail

IMAGE="${1:?usage: dprod-pipeline-test.sh <t1-lartpc.sif> [events]}"
EVENTS="${2:-5}"

if [[ "${T1_SKIP_PIPELINE_TEST:-0}" == "1" ]]; then
    echo "dprod-pipeline-test: SKIPPED (T1_SKIP_PIPELINE_TEST=1)" >&2
    exit 0
fi
if ! nvidia-smi -L > /dev/null 2>&1; then
    echo "dprod-pipeline-test: no GPU on this node; the JAXTPC stages need one." >&2
    echo "  Run on a GPU node, or set T1_SKIP_PIPELINE_TEST=1 to skip explicitly." >&2
    exit 1
fi
: "${LSCRATCH:?dprod-pipeline-test: \$LSCRATCH is not set}"

OUTDIR="$LSCRATCH/t1-software-build/dprod-install-test-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$(dirname "$OUTDIR")"

# ~/.local must not leak into the run: everything a stage imports has to
# come from the image. $LSCRATCH is bound because dprod's job work dirs and
# python's TMPDIR live there.
export APPTAINERENV_PYTHONNOUSERSITE=1
if apptainer exec --nv -B "$LSCRATCH" "$IMAGE" \
        dprod-install-test \
        --site /opt/t1-lartpc-tests/dprod_site_image.yaml \
        --events "$EVENTS" --outdir "$OUTDIR" --work-root "$OUTDIR/work"; then
    rm -rf "$OUTDIR"
    echo "dprod-pipeline-test: PASSED ($IMAGE)"
else
    echo "dprod-pipeline-test: FAILED -- per-stage outputs and logs kept in $OUTDIR (REPORT.txt)" >&2
    exit 1
fi
