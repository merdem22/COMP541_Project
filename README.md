# COMP541 Project

Starter repo for nuScenes-mini exploration plus a minimal lidar/camera fusion baseline with a plug-in graph module.

## Layout
- `scripts/`: quick utilities  
  - `data_sanity_check.py`: load 1–2 batches, print shapes, save BEV plots.  
  - `prepare_data.py`: write token lists for lightweight experiments.  
  - `train_baseline.py`: tiny training loop to validate the pipeline.  
  - `download_nuscenes_mini.py`: checks whether the dataset is present.
- `src/data/`: nuScenes dataset wrapper + collate.
- `src/models/`: lidar backbone, camera backbone, fusion layer, stubbed graph module, detection head.
- `src/utils/`: config + logging helpers.
- `experiments/`: YAML configs (start with `exp_001_baseline_mini.yaml`).
- `docs/`: notes, meeting logs.
- `notebooks/`: exploratory notebooks (nuScenes mini tour).

## Setup
- Create a venv and install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Download **nuScenes-mini** from https://www.nuscenes.org/download (license-restricted) and extract to `data/nuscenes/` so it contains `samples/`, `sweeps/`, `maps/`, and `v1.0-mini/`. `scripts/download_nuscenes_mini.py` only validates that layout.
- Keep `data/` out of git (see `.gitignore`).

## Quickstart checks
- Dataset sanity: `python scripts/data_sanity_check.py --batches 2 --batch-size 2`  
  Saves BEV + boxes to `outputs/sanity/` and prints point/image counts.
- Prepare token cache (optional): `python scripts/prepare_data.py`.
- Run the toy baseline loop: `python scripts/train_baseline.py --config experiments/exp_001_baseline_mini.yaml`  
  This uses dummy targets (zeros) to confirm shapes/gradients; swap in a real head/targets when ready.

## Notebook
- `notebooks/nuscenes_mini_exploration.ipynb` walks through dataset loading, quick stats (scene/sample/annotation counts, class distribution), and prints example annotations.

## Scope notes
- Start with lidar-only for speed; camera BEV pooling keeps the fusion interface alive until proper calibration/projection is wired in.
- Graph reasoning lives in `src/models/graph_module_placeholder.py` as a no-op until the design is finalized.
