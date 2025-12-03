"""
Quick nuScenes-mini sanity check:
- load 1-2 batches
- print shapes
- optionally save BEV visualizations
"""

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Ensure project root is on PYTHONPATH for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nuscenes_dataset import NuScenesDetectionDataset, collate_nuscenes
from src.models.backbone_lidar import points_to_bev
from src.utils.logging import setup_logging


def plot_bev(bev: torch.Tensor, boxes: List[dict], out_path: Path) -> None:
    bev_np = bev.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(bev_np[1], cmap="gray", origin="lower")
    x_range = (-50.0, 50.0)
    y_range = (-50.0, 50.0)
    resolution = 0.5
    for box in boxes:
        cx, cy, _ = box["translation"]
        w, l, _ = box["size"]
        px = (cx - x_range[0]) / resolution
        py = (cy - y_range[0]) / resolution
        rect = plt.Rectangle(
            (px - l / (2 * resolution), py - w / (2 * resolution)),
            l / resolution,
            w / resolution,
            linewidth=1.0,
            edgecolor="lime",
            facecolor="none",
        )
        plt.gca().add_patch(rect)
    plt.title("BEV occupancy (+ boxes)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="nuScenes-mini data sanity check")
    parser.add_argument("--data-root", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sanity"))
    args = parser.parse_args()

    logger = setup_logging(name="data_sanity")
    dataset = NuScenesDetectionDataset(
        data_root=args.data_root,
        version=args.version,
        load_annotations=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_nuscenes,
    )

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.batches:
            break

        bev_batch = [points_to_bev(p) for p in batch["lidar_points"]]
        print_shapes = {
            "tokens": batch["token"],
            "lidar_points_per_sample": [p.shape for p in batch["lidar_points"]],
            "bev_shape": bev_batch[0].shape if bev_batch else None,
            "num_boxes": [len(b) for b in batch["boxes"]],
            "num_images": [len(imgs) for imgs in batch["images"]],
        }
        logger.info(print_shapes)

        for sample_idx, bev in enumerate(bev_batch):
            boxes = batch["boxes"][sample_idx]
            out_path = args.output_dir / f"batch{batch_idx}_sample{sample_idx}_bev.png"
            plot_bev(bev, boxes, out_path)
            logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
