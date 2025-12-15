"""
Minimal training loop for the fusion baseline.
This is intentionally lightweight and focuses on wiring the data pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from nuscenes.utils.data_classes import Box, Quaternion
from nuscenes.utils.geometry_utils import view_points

# For camera overlays
from nuscenes.nuscenes import NuScenes

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
        load_annotations=True,
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


def map_to_grid(x: torch.Tensor, x_range: Tuple[float, float], resolution: float) -> torch.Tensor:
    return torch.floor((x - x_range[0]) / resolution)


def build_targets(
    batch_boxes,
    grid_shape: Tuple[int, int],
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
    resolution: float = 0.5,
    num_classes: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build simple CenterNet-style targets:
    - heatmap with 1 at box centers
    - box regression (dx, dy, z, w, l, h, yaw) at center cell
    - mask indicating where box regression applies
    """
    h, w = grid_shape
    device = torch.device("cpu")
    heatmap = torch.zeros((len(batch_boxes), num_classes, h, w), device=device)
    box_target = torch.zeros((len(batch_boxes), 7, h, w), device=device)
    box_mask = torch.zeros((len(batch_boxes), 1, h, w), device=device)

    for b, boxes in enumerate(batch_boxes):
        for box in boxes:
            cx, cy, cz = box["translation"]
            gx = map_to_grid(torch.tensor(cx), x_range, resolution)
            gy = map_to_grid(torch.tensor(cy), y_range, resolution)
            gx_i = int(gx.item())
            gy_i = int(gy.item())
            if gx_i < 0 or gx_i >= w or gy_i < 0 or gy_i >= h:
                continue

            # Center heatmap
            heatmap[b, 0, gy_i, gx_i] = 1.0

            # Box regression targets
            cell_cx = x_range[0] + (gx_i + 0.5) * resolution
            cell_cy = y_range[0] + (gy_i + 0.5) * resolution
            dx = (cx - cell_cx) / resolution
            dy = (cy - cell_cy) / resolution
            w_box, l_box, h_box = box["size"]
            yaw = box.get("yaw", 0.0)

            box_target[b, :, gy_i, gx_i] = torch.tensor([dx, dy, cz, w_box, l_box, h_box, yaw])
            box_mask[b, 0, gy_i, gx_i] = 1.0

    return heatmap, box_target, box_mask


def decode_predictions(
    heatmap: torch.Tensor,
    box_map: torch.Tensor,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: float,
    score_thresh: float = 0.3,
    topk: int = 20,
):
    """
    Decode top-K heatmap peaks into box dicts in world (lidar) coordinates.
    """
    B, C, H, W = heatmap.shape
    hm_sig = heatmap.sigmoid()
    flat_scores, flat_indices = torch.topk(hm_sig.view(B, -1), k=topk, dim=1)
    preds = []
    for b in range(B):
        boxes = []
        for score, idx in zip(flat_scores[b], flat_indices[b]):
            if score < score_thresh:
                continue
            idx = idx.item()
            gy = idx // W
            gx = idx % W
            reg = box_map[b, :, gy, gx]
            dx, dy, cz, w_box, l_box, h_box, yaw = reg.tolist()
            cell_cx = x_range[0] + (gx + 0.5) * resolution
            cell_cy = y_range[0] + (gy + 0.5) * resolution
            cx = cell_cx + dx * resolution
            cy = cell_cy + dy * resolution
            boxes.append(
                {
                    "translation": [cx, cy, cz],
                    "size": [w_box, l_box, h_box],
                    "yaw": yaw,
                    "score": float(score),
                }
            )
        preds.append(boxes)
    return preds


def draw_boxes(ax, boxes, x_range, y_range, resolution, color="lime", label=None):
    for box in boxes:
        cx, cy, _ = box["translation"]
        w_box, l_box, _ = box["size"]
        gx = (cx - x_range[0]) / resolution
        gy = (cy - y_range[0]) / resolution
        width = l_box / resolution
        height = w_box / resolution
        rect = plt.Rectangle(
            (gx - width / 2, gy - height / 2),
            width,
            height,
            linewidth=1.0,
            edgecolor=color,
            facecolor="none",
            label=label,
        )
        ax.add_patch(rect)


def save_visualizations(
    bev_batch: torch.Tensor,
    batch_boxes,
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    pred_boxes=None,
    out_dir: Path = Path("outputs/vis"),
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
    resolution: float = 0.5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bev_np = bev_batch.cpu().numpy()
    pred_hm = pred_heatmap.sigmoid().cpu().numpy()
    gt_hm = gt_heatmap.cpu().numpy()

    for i in range(bev_np.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(bev_np[i, 1], cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0].set_title("BEV density + GT boxes")
        draw_boxes(axes[0], batch_boxes[i], x_range, y_range, resolution, color="lime", label="GT")
        if pred_boxes:
            draw_boxes(axes[0], pred_boxes[i], x_range, y_range, resolution, color="red", label="Pred")

        axes[1].imshow(gt_hm[i, 0], cmap="plasma", origin="lower")
        axes[1].set_title("GT heatmap")

        axes[2].imshow(pred_hm[i, 0], cmap="plasma", origin="lower")
        axes[2].set_title("Pred heatmap")

        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        out_path = out_dir / f"sample_{i}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def save_camera_overlays(batch_tokens, dataset: NuScenesDetectionDataset, out_dir: Path = Path("outputs/vis_cam")) -> None:
    """
    Save camera views (CAM_FRONT) with GT boxes rendered by nuscenes-devkit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    nusc: NuScenes = dataset.nusc  # type: ignore[attr-defined]
    for token in batch_tokens:
        sample = nusc.get("sample", token)
        if "CAM_FRONT" not in sample["data"]:
            continue
        cam_token = sample["data"]["CAM_FRONT"]
        out_path = out_dir / f"{token}_cam_front.png"
        nusc.render_sample_data(cam_token, out_path=str(out_path), verbose=False)


def save_camera_pred_overlays(
    batch_tokens,
    pred_boxes_batch,
    dataset: NuScenesDetectionDataset,
    out_dir: Path = Path("outputs/vis_cam_pred"),
) -> None:
    """
    Save camera views (CAM_FRONT) with predicted boxes overlaid (red rectangles).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    nusc: NuScenes = dataset.nusc  # type: ignore[attr-defined]
    for token, pred_boxes in zip(batch_tokens, pred_boxes_batch):
        sample = nusc.get("sample", token)
        if "CAM_FRONT" not in sample["data"]:
            continue
        cam_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        img_path = Path(nusc.dataroot) / cam_sd["filename"]
        img = plt.imread(img_path)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(img)

        for box in pred_boxes:
            b = Box(
                center=box["translation"],
                size=box["size"],
                orientation=Quaternion(axis=[0, 0, 1], angle=box.get("yaw", 0.0)),
            )
            # lidar -> ego (lidar)
            b.rotate(Quaternion(lidar_cs["rotation"]))
            b.translate(np.array(lidar_cs["translation"]))
            # ego -> global
            b.rotate(Quaternion(lidar_pose["rotation"]))
            b.translate(np.array(lidar_pose["translation"]))
            # global -> ego (camera)
            b.translate(-np.array(cam_pose["translation"]))
            b.rotate(Quaternion(cam_pose["rotation"]).inverse)
            # ego -> camera
            b.translate(-np.array(cam_cs["translation"]))
            b.rotate(Quaternion(cam_cs["rotation"]).inverse)

            corners = b.corners()
            if (corners[2, :] <= 0).any():
                continue
            pts = view_points(corners, np.array(cam_cs["camera_intrinsic"]), normalize=True)[:2, :]
            x_min, y_min = pts[0].min(), pts[1].min()
            x_max, y_max = pts[0].max(), pts[1].max()
            ax.add_patch(
                plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
            )

        ax.axis("off")
        fig.tight_layout()
        out_path = out_dir / f"{token}_cam_front_pred.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


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
        num_classes=1,
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

        # Build targets to match output grid
        out_h, out_w = heatmap.shape[-2:]
        in_h, in_w = lidar_bevs.shape[-2:]
        scale_x = in_w / out_w
        scale_y = in_h / out_h
        eff_resolution = 0.5 * scale_x  # assumes square scaling; lidar_bevs built with 0.5m resolution
        target_hm, target_box, target_mask = build_targets(
            batch["boxes"],
            grid_shape=(out_h, out_w),
            x_range=(-50.0, 50.0),
            y_range=(-50.0, 50.0),
            resolution=eff_resolution,
            num_classes=heatmap.shape[1],
        )
        target_hm = target_hm.to(device)
        target_box = target_box.to(device)
        target_mask = target_mask.to(device)

        loss_cls = F.binary_cross_entropy_with_logits(heatmap, target_hm)

        # Apply box regression loss only where mask=1
        mask = target_mask.expand_as(box)
        if mask.sum() > 0:
            loss_box = F.l1_loss(box * mask, target_box * mask)
        else:
            loss_box = torch.tensor(0.0, device=device)

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

    # Quick qualitative visualization on a small eval batch
    model.eval()
    eval_loader = DataLoader(
        dataloader.dataset,
        batch_size=min(2, len(dataloader.dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_nuscenes,
    )
    with torch.no_grad():
        eval_batch = next(iter(eval_loader))
        eval_lidar_bevs = torch.stack([points_to_bev(p) for p in eval_batch["lidar_points"]]).to(device)
        eval_cam_stack = stack_camera_images(eval_batch["images"])
        eval_cam_bev = None
        if eval_cam_stack is not None:
            eval_cam_stack = eval_cam_stack.to(device)
            eval_cam_bev = model.camera_backbone.images_to_bev(
                eval_cam_stack, bev_shape=eval_lidar_bevs.shape[-2:]
            )

        eval_outputs = model(eval_lidar_bevs, camera_bev=eval_cam_bev)
        eval_heatmap = eval_outputs["heatmap"].cpu()
        out_h, out_w = eval_heatmap.shape[-2:]
        in_h, in_w = eval_lidar_bevs.shape[-2:]
        scale_x = in_w / out_w
        eff_resolution = 0.5 * scale_x
        gt_hm, _, _ = build_targets(
            eval_batch["boxes"],
            grid_shape=(out_h, out_w),
            x_range=(-50.0, 50.0),
            y_range=(-50.0, 50.0),
            resolution=eff_resolution,
            num_classes=eval_heatmap.shape[1],
        )
        pred_boxes = decode_predictions(
            eval_heatmap,
            eval_outputs["box"].cpu(),
            x_range=(-50.0, 50.0),
            y_range=(-50.0, 50.0),
            resolution=eff_resolution,
            score_thresh=0.3,
            topk=30,
        )
        save_visualizations(
            bev_batch=eval_lidar_bevs.cpu(),
            batch_boxes=eval_batch["boxes"],
            pred_heatmap=eval_heatmap,
            gt_heatmap=gt_hm,
            pred_boxes=pred_boxes,
            out_dir=Path("outputs/vis"),
            x_range=(-50.0, 50.0),
            y_range=(-50.0, 50.0),
            resolution=0.5,
        )
        save_camera_overlays(eval_batch["token"], dataloader.dataset, out_dir=Path("outputs/vis_cam"))
        save_camera_pred_overlays(eval_batch["token"], pred_boxes, dataloader.dataset, out_dir=Path("outputs/vis_cam_pred"))
        logger.info("Saved qualitative visualizations to outputs/vis/")


if __name__ == "__main__":
    main()
