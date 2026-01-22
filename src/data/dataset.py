"""
nuScenes Dataset Loader - Keyframes Only
Handles camera images + LiDAR point clouds with proper calibration
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pyquaternion import Quaternion

# Camera names in nuScenes
CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

# nuScenes class mapping
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Map detection names to our classes
DETECTION_NAMES = {
    'car': 'car',
    'truck': 'truck',
    'construction_vehicle': 'construction_vehicle',
    'bus': 'bus',
    'trailer': 'trailer',
    'barrier': 'barrier',
    'motorcycle': 'motorcycle',
    'bicycle': 'bicycle',
    'pedestrian': 'pedestrian',
    'traffic_cone': 'traffic_cone',
}


class NuScenesDataset(Dataset):
    """nuScenes dataset for 3D object detection (keyframes only)."""
    
    def __init__(
        self,
        root: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        img_size: Tuple[int, int] = (256, 704),
        point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_points: int = 300000,
        augment: bool = True,
        num_sweeps: int = 1,
        sweep_step: int = 1,
        use_time_lag: bool = False,
        use_camera: bool = True,
        image_norm: bool = True,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        color_jitter: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
    ):
        self.root = root
        self.version = version
        self.split = split
        self.img_size = img_size
        self.point_cloud_range = np.array(point_cloud_range)
        self.max_points = max_points
        self.augment = augment and (split == "train")
        self.num_sweeps = max(1, int(num_sweeps))
        self.sweep_step = max(1, int(sweep_step))
        self.use_time_lag = use_time_lag and self.num_sweeps > 1
        self.use_camera = use_camera
        self.image_norm = image_norm
        self.image_mean = np.array(image_mean, dtype=np.float32).reshape(1, 1, 3)
        self.image_std = np.array(image_std, dtype=np.float32).reshape(1, 1, 3)
        self.color_jitter = color_jitter or {}

        if not self.use_camera:
            # LiDAR-only runs should not waste CPU/GPU memory on full-size dummy images.
            # Keep tensors tiny; the model ignores them when `use_camera=False`.
            self._dummy_images = torch.zeros((len(CAMERAS), 3, 1, 1), dtype=torch.float16)
            self._dummy_intrinsics = torch.zeros((len(CAMERAS), 3, 3), dtype=torch.float32)
            self._dummy_extrinsics = torch.zeros((len(CAMERAS), 4, 4), dtype=torch.float32)
        
        # Load nuScenes
        from nuscenes.nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=root, verbose=False)
        
        # Get sample tokens for split
        self.samples = self._get_split_samples()
        
        # Cache for calibration data
        self._calib_cache = {}
        
        # Optional preprocessing cache
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_split_samples(self) -> List[str]:
        """Get sample tokens for train/val split."""
        from nuscenes.utils.splits import create_splits_scenes
        
        splits = create_splits_scenes()
        scene_names = splits[self.split] if self.split in splits else splits['train']
        
        samples = []
        for scene in self.nusc.scene:
            if scene['name'] in scene_names:
                sample_token = scene['first_sample_token']
                while sample_token:
                    samples.append(sample_token)
                    sample = self.nusc.get('sample', sample_token)
                    sample_token = sample['next']
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # Load data
        points, lidar_T_global_from_ego = self._load_lidar(sample)
        if self.use_camera:
            images, cam_intrinsics, cam_extrinsics, cam_T_global_from_ego = self._load_cameras(sample)
        else:
            images = self._dummy_images
            cam_intrinsics = self._dummy_intrinsics
            cam_extrinsics = self._dummy_extrinsics
            cam_T_global_from_ego = torch.zeros((len(CAMERAS), 4, 4), dtype=torch.float32)
        annotations = self._load_annotations(sample)
        
        # Apply augmentation
        if self.augment:
            points, annotations = self._augment(points, annotations)
        
        # Filter points to range
        points = self._filter_points(points)
        
        return {
            'points': torch.from_numpy(points).float(),
            'images': images,  # (6, 3, H, W)
            'cam_intrinsics': cam_intrinsics,  # (6, 3, 3)
            'cam_extrinsics': cam_extrinsics,  # (6, 4, 4)
            # Ego poses per sensor (used by visualization; training can ignore)
            'lidar_T_global_from_ego': torch.from_numpy(lidar_T_global_from_ego).float(),  # (4,4)
            'cam_T_global_from_ego': cam_T_global_from_ego,  # (6,4,4)
            'annotations': annotations,
            'sample_token': sample_token,
        }
    
    def _load_lidar(self, sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load LiDAR point cloud with optional multi-sweep fusion."""
        lidar_token = sample['data']['LIDAR_TOP']
        ref_sd = self.nusc.get('sample_data', lidar_token)
        ref_pose = self.nusc.get('ego_pose', ref_sd['ego_pose_token'])
        ref_T_global_from_ego = self._get_transform(ref_pose['translation'], ref_pose['rotation'])
        ref_T_ego_from_global = np.linalg.inv(ref_T_global_from_ego)

        points_list = []

        sd_token = lidar_token
        sweeps_collected = 0

        while sd_token and sweeps_collected < self.num_sweeps:
            sd = self.nusc.get('sample_data', sd_token)
            pcl_path = os.path.join(self.root, sd['filename'])
            pts = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)

            cs_record = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            ego_pose = self.nusc.get('ego_pose', sd['ego_pose_token'])

            T_ego_from_sensor = self._get_transform(cs_record['translation'], cs_record['rotation'])
            T_global_from_ego = self._get_transform(ego_pose['translation'], ego_pose['rotation'])
            T_global_from_sensor = T_global_from_ego @ T_ego_from_sensor
            T_ref_ego_from_sensor = ref_T_ego_from_global @ T_global_from_sensor

            pts_xyz = pts[:, :3]
            pts_h = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0], 1), dtype=np.float32)], axis=1)
            pts_xyz = (T_ref_ego_from_sensor @ pts_h.T).T[:, :3]
            pts[:, :3] = pts_xyz

            if self.use_time_lag:
                time_lag = (ref_sd['timestamp'] - sd['timestamp']) * 1e-6
                time_col = np.full((pts.shape[0], 1), time_lag, dtype=np.float32)
                pts = np.concatenate([pts, time_col], axis=1)

            points_list.append(pts)
            sweeps_collected += 1

            next_token = sd['prev']
            for _ in range(self.sweep_step - 1):
                if not next_token:
                    break
                next_token = self.nusc.get('sample_data', next_token)['prev']
            sd_token = next_token

        points = np.concatenate(points_list, axis=0) if points_list else np.zeros((0, 5), dtype=np.float32)
        
        # Subsample if needed
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        
        return points, ref_T_global_from_ego
    
    def _load_cameras(self, sample: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load all 6 camera images with calibration."""
        images = []
        intrinsics = []
        extrinsics = []
        ego_poses = []
        
        for cam in CAMERAS:
            cam_token = sample['data'][cam]
            cam_data = self.nusc.get('sample_data', cam_token)
            
            # Load image
            img_path = os.path.join(self.root, cam_data['filename'])
            img = Image.open(img_path).convert('RGB')
            
            # Resize
            orig_size = img.size  # (W, H)
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0

            if self.augment and self.color_jitter.get('enabled', False):
                brightness = self.color_jitter.get('brightness', 0.0)
                contrast = self.color_jitter.get('contrast', 0.0)

                if brightness > 0:
                    factor = 1.0 + np.random.uniform(-brightness, brightness)
                    img = np.clip(img * factor, 0.0, 1.0)
                if contrast > 0:
                    mean = img.mean(axis=(0, 1), keepdims=True)
                    factor = 1.0 + np.random.uniform(-contrast, contrast)
                    img = np.clip((img - mean) * factor + mean, 0.0, 1.0)

            if self.image_norm:
                img = (img - self.image_mean) / self.image_std

            img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
            images.append(img)
            
            # Get calibration
            cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            
            # Intrinsics (adjust for resize)
            K = np.array(cs_record['camera_intrinsic'])
            scale_x = self.img_size[1] / orig_size[0]
            scale_y = self.img_size[0] / orig_size[1]
            K[0] *= scale_x
            K[1] *= scale_y
            intrinsics.append(torch.from_numpy(K).float())
            
            # Extrinsics (sensor to ego)
            trans = np.array(cs_record['translation'])
            rot = Quaternion(cs_record['rotation']).rotation_matrix
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rot
            extrinsic[:3, 3] = trans
            extrinsics.append(torch.from_numpy(extrinsic).float())

            # Ego pose of this camera sample_data (ego -> global)
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_T_global_from_ego = self._get_transform(ego_pose['translation'], ego_pose['rotation'])
            ego_poses.append(torch.from_numpy(ego_T_global_from_ego).float())
        
        return (
            torch.stack(images),
            torch.stack(intrinsics),
            torch.stack(extrinsics),
            torch.stack(ego_poses),
        )
    
    def _load_annotations(self, sample: Dict) -> Dict:
        """Load 3D bounding box annotations."""
        boxes = []
        labels = []
        velocities = []

        lidar_token = sample['data']['LIDAR_TOP']
        ref_sd = self.nusc.get('sample_data', lidar_token)
        ref_pose = self.nusc.get('ego_pose', ref_sd['ego_pose_token'])
        ref_T_global_from_ego = self._get_transform(ref_pose['translation'], ref_pose['rotation'])
        ref_T_ego_from_global = np.linalg.inv(ref_T_global_from_ego)
        ref_R_ego_from_global = ref_T_ego_from_global[:3, :3]
        
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Get class
            det_name = ann['category_name'].split('.')[0]
            if det_name == 'vehicle':
                det_name = ann['category_name'].split('.')[1]
            elif det_name == 'human':
                det_name = 'pedestrian'
            elif det_name == 'movable_object':
                sub = ann['category_name'].split('.')[1]
                if sub == 'trafficcone':
                    det_name = 'traffic_cone'
                elif sub == 'barrier':
                    det_name = 'barrier'
                else:
                    continue
            
            if det_name not in CLASS_TO_IDX:
                continue
            
            # Get box params (center, size, rotation) in reference ego frame
            center_global = np.array(ann['translation'], dtype=np.float32)
            center_h = np.append(center_global, 1.0)
            center = (ref_T_ego_from_global @ center_h)[:3]
            size = np.array(ann['size'])  # w, l, h
            rotation_global = Quaternion(ann['rotation'])
            rot_ego = ref_R_ego_from_global @ rotation_global.rotation_matrix
            # Compute yaw directly to avoid strict orthogonality checks.
            yaw = np.arctan2(rot_ego[1, 0], rot_ego[0, 0])
            
            # Velocity
            vel_global = self.nusc.box_velocity(ann_token)
            if np.any(np.isnan(vel_global)):
                vel_global = np.zeros(3)
            vel_ego = ref_R_ego_from_global @ vel_global
            
            # Box: [x, y, z, w, l, h, yaw]
            boxes.append([center[0], center[1], center[2], 
                         size[0], size[1], size[2], yaw])
            labels.append(CLASS_TO_IDX[det_name])
            velocities.append(vel_ego[:2])
        
        return {
            'boxes': torch.from_numpy(np.array(boxes, dtype=np.float32)) if boxes else torch.zeros((0, 7)),
            'labels': torch.from_numpy(np.array(labels, dtype=np.int64)) if labels else torch.zeros((0,), dtype=torch.long),
            'velocities': torch.from_numpy(np.array(velocities, dtype=np.float32)) if velocities else torch.zeros((0, 2)),
        }
    
    def _transform_points(self, points: np.ndarray, trans: List, rot: Quaternion) -> np.ndarray:
        """Transform points using translation and rotation."""
        points = np.dot(rot.rotation_matrix, points.T).T
        points += np.array(trans)
        return points

    def _get_transform(self, translation: List, rotation: List) -> np.ndarray:
        """Build 4x4 transform matrix from translation and quaternion."""
        rot = Quaternion(rotation).rotation_matrix
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rot
        transform[:3, 3] = np.array(translation, dtype=np.float32)
        return transform
    
    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """Filter points to be within the detection range."""
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) &
            (points[:, 0] <= self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) &
            (points[:, 1] <= self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) &
            (points[:, 2] <= self.point_cloud_range[5])
        )
        return points[mask]
    
    def _augment(self, points: np.ndarray, annotations: Dict) -> Tuple[np.ndarray, Dict]:
        """Apply data augmentation."""
        boxes = annotations['boxes'].numpy() if len(annotations['boxes']) > 0 else np.zeros((0, 7))
        
        # Random flip along Y axis
        if np.random.random() < 0.5:
            points[:, 1] = -points[:, 1]
            if len(boxes) > 0:
                boxes[:, 1] = -boxes[:, 1]
                boxes[:, 6] = -boxes[:, 6]
                annotations['velocities'][:, 1] = -annotations['velocities'][:, 1]
        
        # Random rotation
        rot_angle = np.random.uniform(-0.3925, 0.3925)
        cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        points[:, :2] = np.dot(points[:, :2], rot_mat.T)
        if len(boxes) > 0:
            boxes[:, :2] = np.dot(boxes[:, :2], rot_mat.T)
            boxes[:, 6] += rot_angle
            vel = annotations['velocities'].numpy()
            annotations['velocities'] = torch.from_numpy(np.dot(vel, rot_mat.T)).float()
        
        # Random scaling
        scale = np.random.uniform(0.95, 1.05)
        points[:, :3] *= scale
        if len(boxes) > 0:
            boxes[:, :6] *= scale
        
        annotations['boxes'] = torch.from_numpy(boxes).float()
        return points, annotations


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for variable-size point clouds."""
    # Stack images (all same size)
    images = torch.stack([b['images'] for b in batch])
    cam_intrinsics = torch.stack([b['cam_intrinsics'] for b in batch])
    cam_extrinsics = torch.stack([b['cam_extrinsics'] for b in batch])
    
    # Points need padding
    max_points = max(b['points'].shape[0] for b in batch)
    points_batch = []
    points_mask = []
    
    for b in batch:
        pts = b['points']
        pad_size = max_points - pts.shape[0]
        if pad_size > 0:
            pts = torch.cat([pts, torch.zeros(pad_size, pts.shape[1])], dim=0)
            mask = torch.cat([torch.ones(b['points'].shape[0]), torch.zeros(pad_size)])
        else:
            mask = torch.ones(pts.shape[0])
        points_batch.append(pts)
        points_mask.append(mask)
    
    # Annotations (keep as list for variable number of objects)
    annotations = [b['annotations'] for b in batch]
    
    return {
        'points': torch.stack(points_batch),
        'points_mask': torch.stack(points_mask).bool(),
        'images': images,
        'cam_intrinsics': cam_intrinsics,
        'cam_extrinsics': cam_extrinsics,
        'lidar_T_global_from_ego': torch.stack([b['lidar_T_global_from_ego'] for b in batch]),
        'cam_T_global_from_ego': torch.stack([b['cam_T_global_from_ego'] for b in batch]),
        'annotations': annotations,
        'sample_tokens': [b['sample_token'] for b in batch],
    }
