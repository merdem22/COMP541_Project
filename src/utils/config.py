from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DataConfig:
    data_root: str = "data/nuscenes"
    version: str = "v1.0-mini"
    batch_size: int = 2
    num_workers: int = 2
    shuffle: bool = True
    camera_channels: List[str] = field(
        default_factory=lambda: ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    )


@dataclass
class ModelConfig:
    lidar_in_channels: int = 4
    lidar_bev_channels: int = 2
    lidar_feat_channels: int = 64
    camera_feat_channels: int = 64
    fusion_mode: str = "concat"  # concat | cross_attn
    head_channels: int = 64
    use_graph: bool = False
    graph_k: int = 8


@dataclass
class TrainConfig:
    max_steps: int = 10
    lr: float = 1e-3
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """
    Load YAML config into dataclasses with defaults.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    data_cfg = DataConfig(**(raw.get("data") or {}))
    model_cfg = ModelConfig(**(raw.get("model") or {}))
    train_cfg = TrainConfig(**(raw.get("train") or {}))
    return ExperimentConfig(data=data_cfg, model=model_cfg, train=train_cfg)
