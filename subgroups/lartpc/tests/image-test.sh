#!/usr/bin/env bash
# t1-lartpc image checks: imports, the JAXTPC -> pimm-data and
# DLPGenerator -> edep-sim seam tests, and the vendored unit suites. Runs
# inside the image under either runtime (GPU checks self-skip without one):
#   podman run --rm t1-lartpc-cuda12:<tag> bash /opt/t1-lartpc-tests/image-test.sh
#   apptainer exec [--nv] t1-lartpc-cuda12.sif bash /opt/t1-lartpc-tests/image-test.sh
set -e
python3 -c "\
import ROOT, torch, jax; \
from tools.geometry import generate_detector; \
from tools.simulation import DetectorSimulator; \
import pimm_data, goop, dlp_generator, pysupera, chroma, chroma_lar, photonlib; \
print('JAXTPC + pimm-data + goop (+ chroma-lar/TriChroma) + DLPGenerator + pysupera + PyROOT import OK')"
# The image overrides the base Torch's Triton (see Containerfile): prove the
# base Torch's torch.compile still builds and runs a GPU kernel.
python3 -c "\
import torch, triton; \
ok = not torch.cuda.is_available() or bool((torch.compile(lambda x: torch.sin(x) * 2)(torch.ones(8, device='cuda')) == torch.sin(torch.ones(8, device='cuda')) * 2).all()); \
assert ok, 'torch.compile mismatch'; \
print('torch', torch.__version__, 'triton', triton.__version__, 'torch.compile', 'OK' if torch.cuda.is_available() else 'skipped (no GPU)')"
python3 /opt/t1-lartpc-tests/test_integration.py
# dprod is a checkout used in place (bin/ on PATH), not an installed
# package: prove its CLIs and the pysupera entry point resolve.
dprod --help > /dev/null
dprod-install-test --help > /dev/null
run_pysupera --help > /dev/null
pysupera-app --help > /dev/null

# Re-run each vendored package's own unit suite in-image. All four
# self-skip GPU/asset-dependent cases. JAXTPC's tests resolve config
# fixtures relative to cwd, hence the subshell `cd`.
echo "== JAXTPC unit suite =="
(cd /opt/JAXTPC && JAX_PLATFORM_NAME=cpu python3 -m pytest -q -p no:cacheprovider tests)
echo "== pimm-data unit suite =="
python3 -m pytest -q -p no:cacheprovider /opt/pimm-data/tests
# goop's chroma sampler tests run only with a GPU (`apptainer exec --nv`).
echo "== goop unit suite =="
python3 -m pytest -q -p no:cacheprovider /opt/goop/tests
echo "== pysupera unit suite =="
python3 -m pytest -q -p no:cacheprovider /opt/pysupera/tests

echo "== DLPGenerator + edep-sim seam (bomb-macro -> H5DLP) =="
bash /opt/t1-lartpc-tests/test_generator_edepsim.sh
echo "t1-lartpc image-test: PASSED"
