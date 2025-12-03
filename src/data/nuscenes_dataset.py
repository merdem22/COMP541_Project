import os
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from torch.utils.data import Dataset


class NuScenesDetectionDataset(Dataset):
    """
    Lightweight nuScenes-mini dataset wrapper.

    Returns raw lidar points, camera images, and box metadata without
    binding the downstream model. This keeps the data pipeline stable
    if the modeling code changes later.
    """

    def __init__(
        self,
        data_root: str = "data/nuscenes",
        version: str = "v1.0-mini",
        camera_channels: Sequence[str] = (
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
        ),
        load_lidar: bool = True,
        use_images: bool = True,
        load_annotations: bool = True,
        max_points: int | None = 120000,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.version = version
        self.camera_channels = list(camera_channels)
        self.load_lidar = load_lidar
        self.use_images = use_images
        self.load_annotations = load_annotations
        self.max_points = max_points

        self.nusc = NuScenes(
            version=version,
            dataroot=data_root,
            verbose=verbose,
        )
        self.sample_tokens: List[str] = [s["token"] for s in self.nusc.sample]

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> Dict:
        token = self.sample_tokens[idx]
        sample = self.nusc.get("sample", token)

        lidar_points = torch.empty(0, 4)
        boxes: List[Dict] = []
        if self.load_lidar:
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_points, boxes = self._load_lidar_and_boxes(lidar_token)

        images: Dict[str, torch.Tensor] = {}
        if self.use_images:
            images = self._load_images(sample)

        return {
            "token": token,
            "lidar_points": lidar_points,
            "boxes": boxes,
            "images": images,
        }

    def _load_lidar_and_boxes(self, sample_data_token: str) -> tuple[torch.Tensor, List[Dict]]:
        sd_record = self.nusc.get("sample_data", sample_data_token)
        lidar_path = os.path.join(self.data_root, sd_record["filename"])
        point_cloud = LidarPointCloud.from_file(lidar_path)  # shape (4, N)
        points = torch.from_numpy(point_cloud.points.T).float()  # (N, 4)

        if self.max_points and points.shape[0] > self.max_points:
            choice = torch.randperm(points.shape[0])[: self.max_points]
            points = points[choice]

        boxes: List[Dict] = []
        if self.load_annotations:
            _, box_list, _ = self.nusc.get_sample_data(
                sample_data_token,
                use_flat_vehicle_coordinates=True,
            )
            boxes = [self._box_to_dict(box) for box in box_list]

        return points, boxes

    def _load_images(self, sample: Dict) -> Dict[str, torch.Tensor]:
        images: Dict[str, torch.Tensor] = {}
        for channel in self.camera_channels:
            if channel not in sample["data"]:
                continue
            cam_token = sample["data"][channel]
            cam_sd = self.nusc.get("sample_data", cam_token)
            img_path = os.path.join(self.data_root, cam_sd["filename"])
            with Image.open(img_path) as img:
                img_array = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            images[channel] = tensor
        return images

    @staticmethod
    def _box_to_dict(box: Box) -> Dict:
        # Keep only fields needed for plotting / light baselines.
        return {
            "translation": box.center.tolist(),  # (x, y, z)
            "size": box.wlh.tolist(),  # (width, length, height)
            "rotation": box.orientation.elements.tolist(),  # quaternion (w, x, y, z)
            "name": box.name,
            "token": box.token,
        }


def collate_nuscenes(batch: List[Dict]) -> Dict:
    """
    Collate fn that keeps variable-sized tensors in lists.
    """
    return {
        "token": [item["token"] for item in batch],
        "lidar_points": [item["lidar_points"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "images": [item["images"] for item in batch],
    }
