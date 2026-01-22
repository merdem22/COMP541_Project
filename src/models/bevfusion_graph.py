"""
BEVFusion + Graph Model
Multi-modal 3D Object Detection with Graph Reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .lidar_backbone import LiDARBackbone
from .camera_backbone import CameraBackbone
from .graph_module import GraphModule
from .detection_head import DetectionHead, DetectionLoss, decode_predictions


class BEVFusionGraph(nn.Module):
    """
    BEVFusion-inspired multi-modal 3D detection with Graph Reasoning.
    
    Architecture:
    1. LiDAR Branch: Pillar-based BEV encoding
    2. Camera Branch: LSS-style BEV projection
    3. BEV Fusion: Channel concatenation + fusion conv
    4. Graph Module: Learnable edge graph reasoning (key innovation) - optional
    5. Detection Head: CenterPoint-style heatmap detection
    """
    
    def __init__(self, config: Dict, use_graph: bool = True, use_camera: bool = True):
        super().__init__()
        
        self.config = config
        self.use_graph = use_graph
        self.use_camera = use_camera
        model_cfg = config['model']
        
        # BEV grid parameters
        self.bev_x_bound = model_cfg['bev_x_bound']
        self.bev_y_bound = model_cfg['bev_y_bound']
        self.bev_z_bound = model_cfg['bev_z_bound']
        
        # LiDAR backbone
        lidar_cfg = model_cfg['lidar']
        self.lidar_backbone = LiDARBackbone(
            in_channels=lidar_cfg['in_channels'],
            voxel_size=lidar_cfg['voxel_size'],
            point_cloud_range=lidar_cfg['point_cloud_range'],
            encoder_channels=lidar_cfg['encoder_channels'],
            out_channels=lidar_cfg['bev_out_channels'],
        )
        
        camera_cfg = model_cfg['camera']
        if use_camera:
            # Camera backbone
            self.camera_backbone = CameraBackbone(
                backbone=camera_cfg['backbone'],
                pretrained=camera_cfg['pretrained'],
                depth_channels=camera_cfg.get('depth_channels', 64),
                depth_bins=camera_cfg['depth_bins'],
                depth_min=camera_cfg['depth_min'],
                depth_max=camera_cfg['depth_max'],
                feat_channels=camera_cfg['feat_channels'],
                bev_x_bound=self.bev_x_bound,
                bev_y_bound=self.bev_y_bound,
                bev_z_bound=self.bev_z_bound,
                out_channels=camera_cfg['bev_out_channels'],
                img_size=tuple(camera_cfg['img_size']),
                downsample=camera_cfg.get('downsample', 32),
            )
        else:
            self.camera_backbone = None
        
        # BEV Fusion layer
        lidar_channels = self.lidar_backbone.out_channels
        camera_channels = self.camera_backbone.out_channels if use_camera else 0
        fused_channels = model_cfg['bev_channels']
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(lidar_channels + camera_channels, fused_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, fused_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )
        
        # Graph module (optional - for ablation studies)
        if use_graph:
            graph_cfg = model_cfg['graph']
            self.graph_module = GraphModule(
                in_channels=graph_cfg['in_channels'],
                hidden_channels=graph_cfg['hidden_channels'],
                out_channels=graph_cfg['out_channels'],
                num_layers=graph_cfg['num_layers'],
                kernel_size=graph_cfg.get('kernel_size', 3),
            )
        else:
            self.graph_module = None
        
        # Detection head
        head_cfg = model_cfg['head']
        self.detection_head = DetectionHead(
            in_channels=head_cfg['in_channels'],
            shared_channels=head_cfg['shared_channels'],
            num_classes=head_cfg['num_classes'],
            common_heads=head_cfg['common_heads'],
            tasks=head_cfg['tasks'],
            bev_x_bound=self.bev_x_bound,
            bev_y_bound=self.bev_y_bound,
        )
        
        # Loss
        self.loss_fn = DetectionLoss(config['training']['loss_weights'])
    
    def forward(
        self,
        points: torch.Tensor,
        points_mask: torch.Tensor,
        images: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        cam_extrinsics: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            points: (B, N, 5) LiDAR points
            points_mask: (B, N) valid point mask
            images: (B, 6, 3, H, W) camera images
            cam_intrinsics: (B, 6, 3, 3) camera intrinsics
            cam_extrinsics: (B, 6, 4, 4) camera extrinsics
        
        Returns:
            List of task predictions
        """
        # LiDAR BEV features
        lidar_bev = self.lidar_backbone(points, points_mask)
        
        if self.use_camera and self.camera_backbone is not None:
            # Camera BEV features
            camera_bev = self.camera_backbone(images, cam_intrinsics, cam_extrinsics)
        else:
            camera_bev = None
        
        # Align spatial dimensions if needed
        if camera_bev is not None and lidar_bev.shape[-2:] != camera_bev.shape[-2:]:
            camera_bev = F.interpolate(
                camera_bev,
                size=lidar_bev.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # Fusion
        if camera_bev is not None:
            fused_input = torch.cat([lidar_bev, camera_bev], dim=1)
        else:
            fused_input = lidar_bev
        fused_bev = self.fusion_conv(fused_input)
        
        # Graph reasoning (if enabled)
        if self.use_graph and self.graph_module is not None:
            graph_bev = self.graph_module(fused_bev)
        else:
            graph_bev = fused_bev
        
        # Detection
        predictions = self.detection_head(graph_bev)
        
        return predictions
    
    def compute_loss(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        annotations: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Compute detection loss."""
        bev_shape = predictions[0]['heatmap'].shape[-2:]
        
        targets = self.detection_head.get_targets(
            annotations, 
            predictions[0]['heatmap'].device,
            bev_shape
        )
        
        return self.loss_fn(predictions, targets)
    
    def predict(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        score_thresh: float = 0.1,
        nms_thresh: float = 0.2,
        max_dets: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode predictions to 3D boxes."""
        return decode_predictions(
            predictions,
            self.bev_x_bound,
            self.bev_y_bound,
            score_thresh,
            nms_thresh,
            max_dets,
        )


class BEVFusionGraphLite(nn.Module):
    """
    Lightweight version for faster training/inference.
    Uses smaller backbones and simplified graph module.
    """
    
    def __init__(self, config: Dict, use_graph: bool = True, use_camera: bool = True):
        super().__init__()
        
        self.config = config
        self.use_graph = use_graph
        self.use_camera = use_camera
        model_cfg = config['model']
        
        # BEV parameters
        self.bev_x_bound = model_cfg['bev_x_bound']
        self.bev_y_bound = model_cfg['bev_y_bound']
        self.bev_z_bound = model_cfg['bev_z_bound']
        
        # Simplified LiDAR backbone
        self.lidar_backbone = LiDARBackbone(
            in_channels=5,
            voxel_size=[0.4, 0.4, 8.0],
            point_cloud_range=model_cfg['lidar']['point_cloud_range'],
            encoder_channels=[64, 128],
            out_channels=128,
        )
        
        if use_camera:
            # Simplified camera backbone (no depth prediction, just pooling)
            self.camera_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((32, 32)),
            )
        else:
            self.camera_encoder = None
        
        # Simple BEV projection for camera
        bev_h = int((model_cfg['bev_y_bound'][1] - model_cfg['bev_y_bound'][0]) / model_cfg['bev_y_bound'][2])
        bev_w = int((model_cfg['bev_x_bound'][1] - model_cfg['bev_x_bound'][0]) / model_cfg['bev_x_bound'][2])
        
        if use_camera:
            self.camera_to_bev = nn.Sequential(
                nn.Conv2d(128 * 6, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(bev_h, bev_w), mode='bilinear', align_corners=False),
                nn.Conv2d(256, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        else:
            self.camera_to_bev = None
        
        # Fusion
        fusion_in_channels = 128 + (128 if use_camera else 0)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Simplified graph module (just local attention) - optional
        if use_graph:
            from .graph_module import DenseGraphModule
            self.graph_module = DenseGraphModule(
                in_channels=256,
                hidden_channels=256,
                out_channels=256,
                num_layers=2,
                kernel_size=5,
                num_heads=4,
            )
        else:
            self.graph_module = None
        
        # Detection head
        head_cfg = model_cfg['head']
        self.detection_head = DetectionHead(
            in_channels=256,
            shared_channels=256,
            num_classes=head_cfg['num_classes'],
            common_heads=head_cfg['common_heads'],
            tasks=head_cfg['tasks'],
            bev_x_bound=self.bev_x_bound,
            bev_y_bound=self.bev_y_bound,
        )
        
        self.loss_fn = DetectionLoss(config['training']['loss_weights'])
    
    def forward(
        self,
        points: torch.Tensor,
        points_mask: torch.Tensor,
        images: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        cam_extrinsics: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Forward pass."""
        B = points.shape[0]
        
        # LiDAR BEV
        lidar_bev = self.lidar_backbone(points, points_mask)
        
        # Camera features (simplified)
        B, N_cams, C, H, W = images.shape
        images_flat = images.view(B * N_cams, C, H, W)
        if self.use_camera and self.camera_encoder is not None:
            cam_feat = self.camera_encoder(images_flat)  # (B*6, 128, 32, 32)
            cam_feat = cam_feat.view(B, N_cams * 128, 32, 32)  # (B, 768, 32, 32)
            camera_bev = self.camera_to_bev(cam_feat)
        else:
            camera_bev = None
        
        # Align if needed
        if camera_bev is not None and lidar_bev.shape[-2:] != camera_bev.shape[-2:]:
            camera_bev = F.interpolate(
                camera_bev,
                size=lidar_bev.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Fusion
        if camera_bev is not None:
            fused = torch.cat([lidar_bev, camera_bev], dim=1)
        else:
            fused = lidar_bev
        fused = self.fusion_conv(fused)
        
        # Graph reasoning (if enabled)
        if self.use_graph and self.graph_module is not None:
            graph_out = self.graph_module(fused)
        else:
            graph_out = fused
        
        # Detection
        return self.detection_head(graph_out)
    
    def compute_loss(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        annotations: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Compute loss."""
        bev_shape = predictions[0]['heatmap'].shape[-2:]
        targets = self.detection_head.get_targets(
            annotations,
            predictions[0]['heatmap'].device,
            bev_shape
        )
        return self.loss_fn(predictions, targets)
    
    def predict(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        score_thresh: float = 0.1,
        nms_thresh: float = 0.2,
        max_dets: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode predictions."""
        return decode_predictions(
            predictions,
            self.bev_x_bound,
            self.bev_y_bound,
            score_thresh,
            nms_thresh,
            max_dets,
        )


def build_model(
    config: Dict,
    lite: bool = False,
    use_graph: bool = True,
    use_camera: bool = True,
) -> nn.Module:
    """Build model from config.
    
    Args:
        config: Model configuration dict
        lite: Use lightweight model variant
        use_graph: Whether to use the graph module (for ablation studies)
        use_camera: Whether to use the camera branch (LiDAR-only if False)
    """
    if lite:
        return BEVFusionGraphLite(config, use_graph=use_graph, use_camera=use_camera)
    return BEVFusionGraph(config, use_graph=use_graph, use_camera=use_camera)
