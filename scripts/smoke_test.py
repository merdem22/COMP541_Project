#!/usr/bin/env python3
"""
Quick smoke test to catch obvious data/model regressions.

Runs a couple of batches through dataset -> collate -> model -> loss -> backward.
Designed to be lightweight and safe to run on login/interactive nodes.
"""

import argparse
import os
import sys

import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset import NuScenesDataset, collate_fn  # noqa: E402
from src.models.bevfusion_graph import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp1_lidar.yaml")
    parser.add_argument("--root", type=str, default="nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    use_camera = bool(config["model"].get("use_camera", True))
    use_graph = bool(config["model"].get("use_graph", True))

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    lidar_cfg = model_cfg.get("lidar", {})

    ds = NuScenesDataset(
        root=args.root,
        version=args.version,
        split=args.split,
        img_size=tuple(model_cfg.get("camera", {}).get("img_size", [256, 704])),
        point_cloud_range=lidar_cfg.get(
            "point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        ),
        max_points=int(data_cfg.get("max_points", 200000)),
        augment=False,
        num_sweeps=int(data_cfg.get("num_sweeps", 1)),
        sweep_step=int(data_cfg.get("sweep_step", 1)),
        use_time_lag=bool(data_cfg.get("use_time_lag", False)),
        use_camera=use_camera,
        image_norm=bool(data_cfg.get("image_norm", True)),
        image_mean=tuple(data_cfg.get("image_mean", [0.485, 0.456, 0.406])),
        image_std=tuple(data_cfg.get("image_std", [0.229, 0.224, 0.225])),
        color_jitter=data_cfg.get("color_jitter", None),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, lite=args.lite, use_graph=use_graph, use_camera=use_camera).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_batches = min(args.batches, max(1, len(ds) // args.batch_size))
    for batch_idx in range(num_batches):
        samples = [ds[i] for i in range(batch_idx * args.batch_size, (batch_idx + 1) * args.batch_size)]
        batch = collate_fn(samples)

        points = batch["points"].to(device)
        points_mask = batch["points_mask"].to(device)
        images = batch["images"].to(device)
        cam_intrinsics = batch["cam_intrinsics"].to(device)
        cam_extrinsics = batch["cam_extrinsics"].to(device)

        annotations = []
        for ann in batch["annotations"]:
            annotations.append(
                {
                    "boxes": ann["boxes"].to(device),
                    "labels": ann["labels"].to(device),
                    "velocities": ann["velocities"].to(device),
                }
            )

        optimizer.zero_grad(set_to_none=True)
        preds = model(points, points_mask, images, cam_intrinsics, cam_extrinsics)
        loss_dict = model.compute_loss(preds, annotations)
        loss = loss_dict["total"]
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        print(
            f"[{batch_idx+1}/{num_batches}] loss={loss_value:.4f} "
            f"points={tuple(points.shape)} images={tuple(images.shape)} device={device.type}"
        )


if __name__ == "__main__":
    main()
