# COMP541 Project

Starter repo for exploring nuScenes-mini and prototyping a detection baseline with potential GNN extensions.

## Getting started
- Create a virtualenv and install dependencies you need. For the notebook below: `pip install nuscenes-devkit matplotlib seaborn tqdm`.
- Download **nuScenes-mini** from https://www.nuscenes.org/download and extract to `data/nuscenes/` so it contains `samples/`, `sweeps/`, `maps/`, and `v1.0-mini/`.
- Open the notebook `notebooks/nuscenes_mini_exploration.ipynb` to inspect scenes, annotations, and sketch task ideas.

### Data layout check
- Expected structure for detection work (lidar/camera):  
  `data/nuscenes/{samples/, sweeps/, maps/, v1.0-mini/}`  
  If you only have panoptic labels (e.g., `nuScenes-panoptic-v1.0-mini/panoptic`), also download the main mini split to get point clouds and camera frames.
- Panoptic labels (optional): `data/nuscenes-panoptic/{v1.0-mini/, panoptic/v1.0-mini/}`.
- Keep the `data/` directory out of git (see `.gitignore`).

### Virtualenv
- Recommended to keep a project-local venv (e.g., `python -m venv .venv && source .venv/bin/activate`). `.venv/` is ignored in git.

## Notebook
- `notebooks/nuscenes_mini_exploration.ipynb` walks through dataset loading, quick stats (scene/sample/annotation counts, class distribution), and prints example annotations.
- It ends with a short list of candidate GNN integration points (spatial graph over boxes, temporal fusion across sweeps) to keep the project scope manageable.

## Next steps (suggested)
- Decide final task scope (3D lidar-only vs. camera+lidar) and pick a baseline detector (PointPillars/CenterPoint/SECOND) before adding a GNN block.
- Use nuScenes-mini for iteration and sanity checks; switch to full nuScenes for any serious training/metrics runs.
