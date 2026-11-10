#!/bin/bash
# Slurm job for one nightly build. poll and retry submit it with every
# Slurm option on the sbatch command line, so this file has no #SBATCH lines.
#   build.sbatch <uv> <nightly.py> run <key> --state <dir>
set -euo pipefail
uv="$1"; shift
exec "$uv" run --script "$@"
