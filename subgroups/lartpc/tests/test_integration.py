"""
T1-LArTPC container smoke test: proves the JAXTPC -> pimm-data HDF5 seam
that any data-pipeline or model-eval step built on this image will depend
on -- JAXTPC's simulated deposits, written through its own step-file HDF5
layout, must be readable back by pimm-data's JAXTPCDataset unchanged.

Runs entirely on CPU (JAX_PLATFORM_NAME=cpu, forced below) with five
hand-placed deposits -- no GPU, no out.h5 fixture file, no network.
Deterministic: same deposits in, same signals out, every run.

    apptainer exec --nv images/t1-lartpc-cuda12-latest.sif \\
        python3.11 /opt/t1-lartpc-tests/test_integration.py


Grounded against JAXTPC commit 806f9b3cced279314a3028b9bb4afba4104eca71's
own tests/test_pipeline.py::TestSyntheticPipeline and production/save.py,
and pimm-data commit 168bf0205ba661952ce7e29b67fce76234df8a90's
src/pimm_data/readers/jaxtpc_step.py.
"""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import sys
import tempfile

import numpy as np

sys.path.insert(0, '/opt/JAXTPC')

CONFIG_PATH = '/opt/JAXTPC/config/cubic_wireplane_config.yaml'


def test_step_roundtrip():
    from tools.geometry import generate_detector
    from tools.simulation import DetectorSimulator
    from tools.loader import build_deposit_data

    import h5py
    from production.save import save_event_step, write_config_step
    from pimm_data.jaxtpc import JAXTPCDataset

    # -- 1. Simulate five hand-placed deposits (all in volume 0 / east) --
    det = generate_detector(CONFIG_PATH)
    sim = DetectorSimulator(
        det, total_pad=500, response_chunk_size=500,
        include_track_hits=False, include_noise=False,
        include_electronics=False, include_digitize=False)

    positions_mm = np.array([
        [-500.0, 0.0, 0.0],
        [-800.0, 0.0, 0.0],
        [-1200.0, 0.0, 0.0],
        [-500.0, 100.0, 100.0],
        [-500.0, -100.0, -100.0],
    ], dtype=np.float32)
    de = np.full(5, 2.0, dtype=np.float32)
    dx = np.full(5, 0.3, dtype=np.float32)

    deposits = build_deposit_data(positions_mm, de, dx, sim.config)
    _signals, _track_hits, filled_deposits = sim.process_event(deposits)

    n_east = int(filled_deposits.volumes[0].n_actual)
    n_west = int(filled_deposits.volumes[1].n_actual)
    assert n_east == 5, f"expected all 5 deposits in volume 0, got {n_east}"
    assert n_west == 0, f"expected 0 deposits in volume 1, got {n_west}"

    # -- 2. Write the JAXTPC step-file HDF5 layout pimm-data reads -- 
    with tempfile.TemporaryDirectory() as tmp:
        step_dir = os.path.join(tmp, 'step')
        os.makedirs(step_dir)
        step_path = os.path.join(step_dir, 'sim_step_0000.h5')

        with h5py.File(step_path, 'w') as f:
            write_config_step(
                f, sim.config, dataset_name='sim', file_index=0,
                source_file='synthetic', n_events=1, global_offset=0,
                group_size=5, gap_threshold_mm=5.0)
            save_event_step(
                f, 'event_000', filled_deposits, source_event_idx=0,
                cfg=sim.config)

        # -- 3. Read it back through pimm-data's public loader API -- 
        ds = JAXTPCDataset(data_root=tmp, split='', modalities=('step',))
        assert len(ds) == 1, f"expected 1 event indexed, got {len(ds)}"

        sample = ds.get_data(0)
        step = sample['step']
        n_loaded = step['coord'].shape[0]
        assert n_loaded == n_east + n_west, (
            f"pimm-data loaded {n_loaded} deposits, "
            f"JAXTPC produced {n_east + n_west}")
        assert step['coord'].shape[1] == 3
        assert np.all(np.isfinite(step['energy']))

    print(f"OK: JAXTPC -> pimm-data step seam round-tripped {n_loaded} "
          f"deposits ({n_east} east / {n_west} west)")


if __name__ == '__main__':
    test_step_roundtrip()
