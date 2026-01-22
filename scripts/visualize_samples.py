#!/usr/bin/env python3
"""
Simple, robust visualization for nuScenes BEV detectors.

Outputs:
  - BEV plot (LiDAR XY + GT boxes + predicted boxes)
  - 6 per-camera images with projected 3D boxes (GT green, preds red)

The script selects top samples from a random subset using a simple
distance-based F1 proxy (same matching idea as training metrics).
"""

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import CLASS_NAMES  # noqa: E402
from src.data.dataset import CAMERAS, NuScenesDataset, collate_fn  # noqa: E402
from src.models.bevfusion_graph import build_model  # noqa: E402


@dataclass
class SampleScore:
    idx: int
    f1: float
    precision: float
    recall: float
    num_pred: int
    num_gt: int


def _filter_by_score(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    score_thresh: float,
    max_dets: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scores.size == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    keep = scores >= float(score_thresh)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if max_dets is not None and scores.size > int(max_dets):
        order = np.argsort(-scores)[: int(max_dets)]
        boxes = boxes[order]
        scores = scores[order]
        labels = labels[order]
    return boxes, scores, labels


def _box_corners_xy(box: np.ndarray) -> np.ndarray:
    x, y, _, w, l, _, yaw = box.tolist()
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    dx = l / 2.0
    dy = w / 2.0
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float32) @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def _box_corners_3d_ego(box: np.ndarray) -> np.ndarray:
    x, y, z, w, l, h, yaw = box.tolist()
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    corners = np.array(
        [
            [dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],
            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
        ],
        dtype=np.float32,
    )
    corners = corners @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z
    return corners


def _ego_to_cam(pts_ego: np.ndarray, T_ego_from_cam: np.ndarray) -> np.ndarray:
    """Convert ego-frame points to camera frame given T_ego_from_cam (cam->ego)."""
    R = T_ego_from_cam[:3, :3]
    t = T_ego_from_cam[:3, 3]
    # ego = R * cam + t  => cam = R^T * (ego - t)
    # For row-vector points: (R^T @ v)^T == v^T @ R  => (ego - t) @ R
    return (pts_ego - t[None, :]) @ R


def _transform_points_h(pts_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0], 1), dtype=np.float32)], axis=1)
    out = (T @ pts_h.T).T
    return out[:, :3].astype(np.float32)


def _project(pts_cam: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points (N,3) to pixels (N,2). Assumes z forward."""
    z = pts_cam[:, 2]
    valid = z > 1e-3
    uv = np.full((pts_cam.shape[0], 2), np.nan, dtype=np.float32)
    pts = pts_cam[valid]
    if pts.shape[0] == 0:
        return uv, valid
    u = K[0, 0] * (pts[:, 0] / pts[:, 2]) + K[0, 2]
    v = K[1, 1] * (pts[:, 1] / pts[:, 2]) + K[1, 2]
    uv[valid] = np.stack([u, v], axis=1).astype(np.float32)
    return uv, valid


def _to_uint8(img: torch.Tensor, image_norm: bool, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> np.ndarray:
    """(3,H,W) -> uint8 HxWx3, robust to normalization mismatches."""
    x = img.detach().cpu().float()
    x_min = float(x.min().item()) if x.numel() else 0.0
    x_max = float(x.max().item()) if x.numel() else 0.0
    x = x.permute(1, 2, 0).numpy()

    if image_norm and (x_min < -0.5 or x_max > 1.5):
        x = x * np.array(std, dtype=np.float32)[None, None, :] + np.array(mean, dtype=np.float32)[None, None, :]
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)

    if x_max > 2.0:
        return np.clip(x, 0.0, 255.0).astype(np.uint8)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def _resize_nearest(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    yy = (np.linspace(0, h - 1, nh)).astype(np.int32)
    xx = (np.linspace(0, w - 1, nw)).astype(np.int32)
    return img[yy][:, xx]


def _match_distance(pred_xy: np.ndarray, gt_xy: np.ndarray, thresh: float) -> Tuple[int, int, int]:
    if pred_xy.shape[0] == 0:
        return 0, 0, int(gt_xy.shape[0])
    if gt_xy.shape[0] == 0:
        return 0, int(pred_xy.shape[0]), 0

    matched = np.zeros((gt_xy.shape[0],), dtype=bool)
    tp, fp = 0, 0
    for p in pred_xy:
        d = np.sqrt(((gt_xy - p[None, :]) ** 2).sum(axis=1))
        j = int(d.argmin())
        if d[j] < thresh and not matched[j]:
            matched[j] = True
            tp += 1
        else:
            fp += 1
    fn = int((~matched).sum())
    return tp, fp, fn


def _match_tp_indices(
    pred_xy: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_xy: np.ndarray,
    gt_labels: np.ndarray,
    distance_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy matching in BEV by class using center distance threshold.
    Returns:
      - tp_pred_idx: indices into preds that are matched (TP)
      - tp_gt_idx: indices into GT that are matched
    """
    if pred_xy.shape[0] == 0 or gt_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    order = np.argsort(-pred_scores) if pred_scores.size else np.arange(pred_xy.shape[0], dtype=np.int64)
    gt_matched = np.zeros((gt_xy.shape[0],), dtype=bool)
    tp_pred = []
    tp_gt = []

    for i in order:
        cls = int(pred_labels[i])
        gt_mask = (gt_labels == cls) & (~gt_matched)
        if not np.any(gt_mask):
            continue
        gt_idx = np.where(gt_mask)[0]
        d = np.sqrt(((gt_xy[gt_idx] - pred_xy[i][None, :]) ** 2).sum(axis=1))
        j_local = int(d.argmin())
        if d[j_local] < distance_thresh:
            j = int(gt_idx[j_local])
            gt_matched[j] = True
            tp_pred.append(int(i))
            tp_gt.append(int(j))

    return np.array(tp_pred, dtype=np.int64), np.array(tp_gt, dtype=np.int64)


def _score_sample(decoded: Dict[str, torch.Tensor], ann: Dict[str, torch.Tensor], distance_thresh: float) -> SampleScore:
    pred_xy = decoded["boxes"][:, :2].detach().cpu().numpy()
    pred_labels = decoded["labels"].detach().cpu().numpy().astype(np.int64)
    gt_xy = ann["boxes"][:, :2].detach().cpu().numpy()
    gt_labels = ann["labels"].detach().cpu().numpy().astype(np.int64)

    tp = fp = fn = 0
    for cls_idx in range(len(CLASS_NAMES)):
        p = pred_xy[pred_labels == cls_idx]
        g = gt_xy[gt_labels == cls_idx]
        tpi, fpi, fni = _match_distance(p, g, distance_thresh)
        tp += tpi
        fp += fpi
        fn += fni

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return SampleScore(
        idx=-1,
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        num_pred=int(pred_xy.shape[0]),
        num_gt=int(gt_xy.shape[0]),
    )


def _save_bev(points: np.ndarray, gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray, pred_labels: np.ndarray, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if points.shape[0] > 0:
        ax.scatter(points[:, 0], points[:, 1], s=0.2, c=points[:, 2], cmap="viridis", alpha=0.6)

    for b in gt_boxes:
        corners = _box_corners_xy(b)
        poly = np.vstack([corners, corners[0]])
        ax.plot(poly[:, 0], poly[:, 1], color="lime", linewidth=1.2)

    order = np.argsort(-pred_scores) if pred_scores.size else np.array([], dtype=np.int64)
    for j in order[:50]:
        b = pred_boxes[j]
        corners = _box_corners_xy(b)
        poly = np.vstack([corners, corners[0]])
        ax.plot(poly[:, 0], poly[:, 1], color="red", linewidth=1.0)
        ax.text(float(b[0]), float(b[1]), f"{CLASS_NAMES[int(pred_labels[j])]}:{pred_scores[j]:.2f}", color="red", fontsize=6)

    ax.set_title(title)
    ax.set_xlim([-55, 55])
    ax.set_ylim([-55, 55])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_cameras(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    lidar_T_global_from_ego: np.ndarray,
    cam_T_global_from_ego: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    out_dir: str,
    prefix: str,
    image_norm: bool,
    image_mean: Tuple[float, float, float],
    image_std: Tuple[float, float, float],
    cam_scale: float,
    score_thresh: float,
    max_pred: int,
) -> None:
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    # Filter preds for clarity
    if pred_scores.size:
        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        order = np.argsort(-pred_scores)[:max_pred]
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]
        pred_labels = pred_labels[order]

    for cam_idx in range(min(int(images.shape[0]), len(CAMERAS))):
        img_u8 = _to_uint8(images[cam_idx], image_norm, image_mean, image_std)
        K = intrinsics[cam_idx].detach().cpu().numpy().copy()
        T = extrinsics[cam_idx].detach().cpu().numpy()
        T_global_from_ego_cam = cam_T_global_from_ego[cam_idx]
        T_ego_cam_from_global = np.linalg.inv(T_global_from_ego_cam)
        T_global_from_ego_lidar = lidar_T_global_from_ego

        if cam_scale != 1.0:
            img_u8 = _resize_nearest(img_u8, cam_scale)
            K[0, 0] *= cam_scale
            K[1, 1] *= cam_scale
            K[0, 2] *= cam_scale
            K[1, 2] *= cam_scale

        img = Image.fromarray(img_u8)
        draw = ImageDraw.Draw(img)

        def draw_box(box: np.ndarray, color: Tuple[int, int, int]) -> None:
            corners = _box_corners_3d_ego(box)
            # Align from LiDAR ego frame -> global -> this camera's ego frame -> camera frame
            corners_global = _transform_points_h(corners, T_global_from_ego_lidar)
            corners_ego_cam = _transform_points_h(corners_global, T_ego_cam_from_global)
            uv, valid = _project(_ego_to_cam(corners_ego_cam, T), K)
            for a, b in edges:
                if not (valid[a] and valid[b]):
                    continue
                x0, y0 = float(uv[a, 0]), float(uv[a, 1])
                x1, y1 = float(uv[b, 0]), float(uv[b, 1])
                if np.any(np.isnan([x0, y0, x1, y1])):
                    continue
                draw.line([(x0, y0), (x1, y1)], fill=color, width=2)

        # GT (green)
        for b in gt_boxes:
            draw_box(b, (0, 255, 0))

        # Pred (red)
        for b, s, lbl in zip(pred_boxes, pred_scores, pred_labels):
            draw_box(b, (255, 0, 0))
            corners = _box_corners_3d_ego(b)
            corners_global = _transform_points_h(corners, T_global_from_ego_lidar)
            corners_ego_cam = _transform_points_h(corners_global, T_ego_cam_from_global)
            uv, valid = _project(_ego_to_cam(corners_ego_cam, T), K)
            if valid.any():
                i0 = int(np.where(valid)[0][0])
                x0, y0 = float(uv[i0, 0]), float(uv[i0, 1])
                if not np.any(np.isnan([x0, y0])):
                    draw.text((x0, y0), f"{CLASS_NAMES[int(lbl)]}:{float(s):.2f}", fill=(255, 0, 0))

        cam_name = CAMERAS[cam_idx]
        out_path = os.path.join(out_dir, f"{prefix}_{cam_name}.png")
        img.save(out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--split", default="val")
    p.add_argument("--num-candidates", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs/vis")
    p.add_argument("--device", default="cuda")
    p.add_argument("--score-thresh", type=float, default=None)
    p.add_argument("--nms-thresh", type=float, default=None)
    p.add_argument("--max-dets", type=int, default=None)
    p.add_argument("--distance-thresh", type=float, default=None)
    p.add_argument(
        "--eval-score-thresh",
        type=float,
        default=None,
        help="Score threshold used for proxy metrics + sample selection (after decoding). Defaults to max(--score-thresh, --cam-score-thresh).",
    )
    p.add_argument("--save-cameras", action="store_true")
    p.add_argument("--cam-scale", type=float, default=1.0)
    p.add_argument("--cam-score-thresh", type=float, default=0.2)
    p.add_argument("--cam-max-pred", type=int, default=30)
    p.add_argument(
        "--save-mode",
        choices=["topk", "demo"],
        default="topk",
        help="topk: save --save-topk best samples. demo: save 1 good + 1 questionable sample.",
    )
    p.add_argument("--save-topk", type=int, default=3, help="Save top-k best samples by proxy F1 (from candidate subset).")
    p.add_argument("--min-f1", type=float, default=0.0, help="Only save samples with proxy F1 >= this.")
    p.add_argument("--min-precision", type=float, default=0.0, help="Only save samples with proxy precision >= this.")
    p.add_argument("--min-recall", type=float, default=0.0, help="Only save samples with proxy recall >= this.")
    p.add_argument(
        "--require-class",
        default="",
        help="Comma-separated class names that must appear as matched TPs in a saved sample (e.g. 'car,pedestrian').",
    )
    p.add_argument(
        "--pred-mode",
        choices=["all", "tp"],
        default="all",
        help="Which predicted boxes to draw: all (after thresholds) or only matched TPs (cleaner demos).",
    )
    p.add_argument(
        "--gt-mode",
        choices=["all", "matched"],
        default="all",
        help="Which GT boxes to draw: all or only those matched to a prediction.",
    )
    p.add_argument(
        "--good-pred-mode",
        choices=["all", "tp"],
        default="tp",
        help="Prediction drawing mode for the 'good' demo sample (only used with --save-mode demo).",
    )
    p.add_argument(
        "--questionable-pred-mode",
        choices=["all", "tp"],
        default="all",
        help="Prediction drawing mode for the 'questionable' demo sample (only used with --save-mode demo).",
    )
    p.add_argument(
        "--questionable-target",
        choices=["low_precision", "median_f1"],
        default="low_precision",
        help="How to pick the 'questionable' demo sample (only used with --save-mode demo).",
    )
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--strict-load", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.no_graph:
        cfg["model"]["use_graph"] = False
    if args.no_camera:
        cfg["model"]["use_camera"] = False
        cfg.setdefault("data", {})
        cfg["data"]["use_camera"] = False

    eval_cfg = cfg.get("eval", {})
    score_thresh = float(args.score_thresh) if args.score_thresh is not None else float(eval_cfg.get("score_thresh", 0.1))
    nms_thresh = float(args.nms_thresh) if args.nms_thresh is not None else float(eval_cfg.get("nms_thresh", 0.2))
    max_dets = int(args.max_dets) if args.max_dets is not None else int(eval_cfg.get("max_dets", 300))
    dist_thresh = float(args.distance_thresh) if args.distance_thresh is not None else float(eval_cfg.get("distance_thresh", 2.0))
    eval_score_thresh = (
        float(args.eval_score_thresh)
        if args.eval_score_thresh is not None
        else float(max(score_thresh, float(args.cam_score_thresh)))
    )

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    model = build_model(
        cfg,
        lite=False,
        use_graph=cfg["model"].get("use_graph", True),
        use_camera=cfg["model"].get("use_camera", True),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        if args.strict_load:
            raise
        model.load_state_dict(state, strict=False)
    model.eval()

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    lidar_cfg = model_cfg.get("lidar", {})
    cam_cfg = model_cfg.get("camera", {})

    ds = NuScenesDataset(
        root=args.root,
        version=args.version,
        split=args.split,
        img_size=tuple(cam_cfg.get("img_size", [256, 512])),
        point_cloud_range=lidar_cfg.get("point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        max_points=int(data_cfg.get("max_points", 200000)),
        augment=False,
        num_sweeps=int(data_cfg.get("num_sweeps", 1)),
        sweep_step=int(data_cfg.get("sweep_step", 1)),
        use_time_lag=bool(data_cfg.get("use_time_lag", False)),
        use_camera=bool(data_cfg.get("use_camera", True)) or bool(args.save_cameras),
        image_norm=bool(data_cfg.get("image_norm", True)),
        image_mean=tuple(data_cfg.get("image_mean", [0.485, 0.456, 0.406])),
        image_std=tuple(data_cfg.get("image_std", [0.229, 0.224, 0.225])),
        color_jitter=data_cfg.get("color_jitter", {}),
    )

    rnd = random.Random(args.seed)
    indices = list(range(len(ds)))
    rnd.shuffle(indices)
    indices = indices[: min(int(args.num_candidates), len(indices))]

    amp = device.type == "cuda"
    decoded_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    scores: List[SampleScore] = []

    for idx in indices:
        sample = ds[idx]
        batch = collate_fn([sample])
        points = batch["points"].to(device)
        mask = batch["points_mask"].to(device)
        images = batch["images"].to(device)
        K = batch["cam_intrinsics"].to(device)
        E = batch["cam_extrinsics"].to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
            preds = model(points, mask, images, K, E)
            dec0 = model.predict(preds, score_thresh=score_thresh, nms_thresh=nms_thresh, max_dets=max_dets)[0]

        decoded_cache[idx] = {k: v.detach().cpu() for k, v in dec0.items()}
        # Score using an additional threshold (more stable + demo-friendly than a very low decode threshold).
        dec_boxes = dec0["boxes"].detach().cpu().numpy() if dec0["boxes"].numel() else np.zeros((0, 7), dtype=np.float32)
        dec_scores = dec0["scores"].detach().cpu().numpy() if dec0["scores"].numel() else np.zeros((0,), dtype=np.float32)
        dec_labels = dec0["labels"].detach().cpu().numpy().astype(np.int64) if dec0["labels"].numel() else np.zeros((0,), dtype=np.int64)
        f_boxes, f_scores, f_labels = _filter_by_score(dec_boxes, dec_scores, dec_labels, eval_score_thresh, max_dets=None)
        dec_eval = {
            "boxes": torch.from_numpy(f_boxes),
            "scores": torch.from_numpy(f_scores),
            "labels": torch.from_numpy(f_labels),
        }
        s = _score_sample(dec_eval, sample["annotations"], dist_thresh)
        s.idx = idx
        scores.append(s)

    scores.sort(key=lambda s: s.f1)
    best_first = list(reversed(scores))  # descending by F1

    require_classes = [c.strip() for c in str(args.require_class).split(",") if c.strip()]
    require_class_ids = []
    if require_classes:
        name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
        unknown = [c for c in require_classes if c not in name_to_idx]
        if unknown:
            raise ValueError(f"Unknown class(es) in --require-class: {unknown}. Valid: {CLASS_NAMES}")
        require_class_ids = [name_to_idx[c] for c in require_classes]

    def _has_required_tp(idx: int) -> bool:
        if not require_class_ids:
            return True
        sample = ds[idx]
        dec = decoded_cache[idx]
        ann = sample["annotations"]
        gt_boxes = ann["boxes"].numpy() if ann["boxes"].numel() else np.zeros((0, 7), dtype=np.float32)
        gt_labels = ann["labels"].numpy().astype(np.int64) if ann["labels"].numel() else np.zeros((0,), dtype=np.int64)
        pred_boxes = dec["boxes"].numpy() if dec["boxes"].numel() else np.zeros((0, 7), dtype=np.float32)
        pred_scores = dec["scores"].numpy() if dec["scores"].numel() else np.zeros((0,), dtype=np.float32)
        pred_labels = dec["labels"].numpy() if dec["labels"].numel() else np.zeros((0,), dtype=np.int64)
        pred_boxes, pred_scores, pred_labels = _filter_by_score(
            pred_boxes,
            pred_scores,
            pred_labels,
            eval_score_thresh,
            max_dets=None,
        )
        tp_pred_idx, _ = _match_tp_indices(
            pred_xy=pred_boxes[:, :2] if pred_boxes.size else np.zeros((0, 2), dtype=np.float32),
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            gt_xy=gt_boxes[:, :2] if gt_boxes.size else np.zeros((0, 2), dtype=np.float32),
            gt_labels=gt_labels,
            distance_thresh=dist_thresh,
        )
        if tp_pred_idx.size == 0:
            return False
        tp_labels = pred_labels[tp_pred_idx]
        return all(np.any(tp_labels == c) for c in require_class_ids)

    def dump(which: str, s: SampleScore) -> None:
        sample = ds[s.idx]
        dec = decoded_cache[s.idx]
        pts = sample["points"].numpy()
        ann = sample["annotations"]

        gt_boxes = ann["boxes"].numpy() if ann["boxes"].numel() else np.zeros((0, 7), dtype=np.float32)
        gt_labels = ann["labels"].numpy().astype(np.int64) if ann["labels"].numel() else np.zeros((0,), dtype=np.int64)
        pred_boxes = dec["boxes"].numpy() if dec["boxes"].numel() else np.zeros((0, 7), dtype=np.float32)
        pred_scores = dec["scores"].numpy() if dec["scores"].numel() else np.zeros((0,), dtype=np.float32)
        pred_labels = dec["labels"].numpy() if dec["labels"].numel() else np.zeros((0,), dtype=np.int64)

        if args.pred_mode == "tp" or args.gt_mode == "matched":
            tp_pred_idx, tp_gt_idx = _match_tp_indices(
                pred_xy=pred_boxes[:, :2] if pred_boxes.size else np.zeros((0, 2), dtype=np.float32),
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                gt_xy=gt_boxes[:, :2] if gt_boxes.size else np.zeros((0, 2), dtype=np.float32),
                gt_labels=gt_labels,
                distance_thresh=dist_thresh,
            )
            if args.pred_mode == "tp":
                pred_boxes = pred_boxes[tp_pred_idx] if tp_pred_idx.size else np.zeros((0, 7), dtype=np.float32)
                pred_scores = pred_scores[tp_pred_idx] if tp_pred_idx.size else np.zeros((0,), dtype=np.float32)
                pred_labels = pred_labels[tp_pred_idx] if tp_pred_idx.size else np.zeros((0,), dtype=np.int64)
            if args.gt_mode == "matched":
                gt_boxes = gt_boxes[tp_gt_idx] if tp_gt_idx.size else np.zeros((0, 7), dtype=np.float32)

        tag = f"{which}_idx{s.idx}_f1{s.f1:.3f}_p{s.precision:.3f}_r{s.recall:.3f}"
        bev_path = os.path.join(args.outdir, f"{tag}_bev.png")
        title = f"{tag} pred={s.num_pred} gt={s.num_gt}"
        _save_bev(pts, gt_boxes, pred_boxes, pred_scores, pred_labels, bev_path, title)

        if args.save_cameras and sample["images"].numel() > 0 and sample["images"].shape[-1] > 1:
            cam_dir = os.path.join(args.outdir, f"{tag}_cams")
            os.makedirs(cam_dir, exist_ok=True)
            _save_cameras(
                images=sample["images"],
                intrinsics=sample["cam_intrinsics"],
                extrinsics=sample["cam_extrinsics"],
                lidar_T_global_from_ego=sample["lidar_T_global_from_ego"].numpy(),
                cam_T_global_from_ego=sample["cam_T_global_from_ego"].numpy(),
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                out_dir=cam_dir,
                prefix=tag,
                image_norm=bool(data_cfg.get("image_norm", True)),
                image_mean=tuple(data_cfg.get("image_mean", [0.485, 0.456, 0.406])),
                image_std=tuple(data_cfg.get("image_std", [0.229, 0.224, 0.225])),
                cam_scale=float(args.cam_scale),
                score_thresh=float(args.cam_score_thresh),
                max_pred=int(args.cam_max_pred),
            )

    eligible = [
        s
        for s in best_first
        if (
            s.f1 >= float(args.min_f1)
            and s.precision >= float(args.min_precision)
            and s.recall >= float(args.min_recall)
            and _has_required_tp(s.idx)
        )
    ]
    if len(eligible) == 0:
        # Fall back: ignore require-class if it was too strict.
        eligible = [
            s
            for s in best_first
            if (s.f1 >= float(args.min_f1) and s.precision >= float(args.min_precision) and s.recall >= float(args.min_recall))
        ]
        if len(eligible) == 0:
            eligible = best_first[: max(1, int(args.save_topk))]

    if args.save_mode == "demo":
        good = eligible[0]

        remaining = [s for s in eligible if s.idx != good.idx]
        questionable = None
        if remaining:
            if args.questionable_target == "low_precision":
                remaining.sort(key=lambda s: (s.precision, -s.recall, -s.f1))
                questionable = remaining[0]
            else:
                mid_f1 = remaining[len(remaining) // 2].f1
                remaining.sort(key=lambda s: (abs(s.f1 - mid_f1), -s.recall, -s.precision))
                questionable = remaining[0]

        # Good: keep visuals clean by showing only matched TPs.
        args.pred_mode = args.good_pred_mode
        dump("good", good)

        if questionable is not None:
            # Questionable: show raw predictions (or TPs if configured).
            args.pred_mode = args.questionable_pred_mode
            dump("questionable", questionable)
            to_save = [good, questionable]
        else:
            to_save = [good]
    else:
        to_save = eligible[: max(1, int(args.save_topk))]
        for rank, s in enumerate(to_save, start=1):
            dump(f"top{rank}", s)

    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        for s in scores:
            f.write(f"{s.idx}\tf1={s.f1:.4f}\tp={s.precision:.4f}\tr={s.recall:.4f}\tpred={s.num_pred}\tgt={s.num_gt}\n")

    print(f"Saved: {args.outdir}")
    print("Saved indices:", ", ".join(str(s.idx) for s in to_save))


if __name__ == "__main__":
    main()
