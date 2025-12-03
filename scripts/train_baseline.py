"""
Minimal training loop for the fusion baseline.
This is intentionally lightweight and focuses on wiring the data pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

# Ensure project root is on PYTHONPATH for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nuscenes_dataset import NuScenesDetectionDataset, collate_nuscenes
from src.models.backbone_lidar import points_to_bev
from src.models.fusion_baseline import FusionBaselineModel
from src.utils.config import load_config
from src.utils.logging import setup_logging


def build_dataloader(cfg) -> DataLoader:
    dataset = NuScenesDetectionDataset(
        data_root=cfg.data.data_root,
        version=cfg.data.version,
        camera_channels=cfg.data.camera_channels,
        load_annotations=False,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_nuscenes,
    )


def stack_camera_images(image_list) -> Optional[torch.Tensor]:
    tensors = []
    for images in image_list:
        if not images:
            return None
        if "CAM_FRONT" in images:
            img = images["CAM_FRONT"]
        else:
            # Fallback to the first available camera channel
            first_key = next(iter(images))
            img = images[first_key]
        tensors.append(img)
    return torch.stack(tensors) if tensors else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny fusion baseline.")
    parser.add_argument("--config", default="experiments/exp_001_baseline_mini.yaml", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(name="train_baseline")
    device = torch.device(cfg.train.device)

    dataloader = build_dataloader(cfg)
    model = FusionBaselineModel(
        lidar_in_channels=cfg.model.lidar_bev_channels,
        lidar_feat_channels=cfg.model.lidar_feat_channels,
        camera_feat_channels=cfg.model.camera_feat_channels,
        fusion_mode=cfg.model.fusion_mode,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    step = 0
    model.train()
    for batch in dataloader:
        lidar_bevs = torch.stack([points_to_bev(p) for p in batch["lidar_points"]]).to(device)
        camera_stack = stack_camera_images(batch["images"])
        camera_bev = None
        if camera_stack is not None:
            camera_stack = camera_stack.to(device)
            camera_bev = model.camera_backbone.images_to_bev(camera_stack, bev_shape=lidar_bevs.shape[-2:])

        outputs = model(lidar_bevs, camera_bev=camera_bev)
        heatmap = outputs["heatmap"]
        box = outputs["box"]
        loss_cls = F.mse_loss(heatmap, torch.zeros_like(heatmap))
        loss_box = F.l1_loss(box, torch.zeros_like(box))
        loss = loss_cls + 0.1 * loss_box

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(
            "step=%d loss=%.4f heatmap_mean=%.4f box_mean=%.4f",
            step,
            loss.item(),
            heatmap.mean().item(),
            box.mean().item(),
        )

        step += 1
        if step >= cfg.train.max_steps:
            break

    logger.info("Finished %d steps", step)


if __name__ == "__main__":
    main()
