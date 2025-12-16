"""
Learned Dynamic Graph Module for Multi-Sensor Fusion.

This module learns which edges should exist between BEV nodes,
replacing the fixed k-NN graph topology with an adaptive, learned one.

Key idea from proposal:
    s_ij = f_θ(h_i, h_j, Δp_ij)
    Keep Top-K edges per node based on learned scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LearnedGraphModule(nn.Module):
    """
    Learns graph topology end-to-end with detection loss.
    
    Unlike StaticGraphModule (fixed 8-neighbor grid), this module:
    1. Computes edge scores for candidate pairs
    2. DYNAMICALLY selects edges based on learned threshold
    3. Performs message passing on the learned sparse graph
    
    Key innovation: The number of edges is LEARNED, not fixed!
    - Edges with score > threshold are kept
    - Threshold is learned during training
    - Different scenes can have different numbers of edges
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        max_edges_per_node: int = 8,  # Maximum edges (for memory efficiency)
        candidate_radius: int = 2,  # Reduced to 2 (24 neighbors) for speed on CPU
        use_position_encoding: bool = True,
        initial_threshold: float = 0.0,  # Neutral start
        temperature: float = 0.1,  # For soft masking gradient flow
    ):
        super().__init__()
        self.in_channels = in_channels
        self.max_edges_per_node = max_edges_per_node
        self.candidate_radius = candidate_radius
        self.use_position_encoding = use_position_encoding
        
        # Edge scoring network: f_θ(h_i, h_j, Δp_ij) → s_ij ∈ [0, 1]
        edge_input_dim = 2 * in_channels + (2 if use_position_encoding else 0)
        self.edge_scorer = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # LEARNABLE threshold for dynamic edge selection
        # This is the raw value BEFORE sigmoid (so sigmoid(initial_threshold) is actual threshold)
        # E.g., initial_threshold=-1.0 → sigmoid(-1)=0.27 → 27% threshold
        self.edge_threshold = nn.Parameter(torch.tensor(float(initial_threshold)))
        
        # Message transformation
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        
        # Output projection (after aggregation)
        self.output_proj = nn.Linear(in_channels, in_channels)
        
        self.temperature = temperature
        
        # For tracking statistics
        self.last_num_edges = 0
        self.last_avg_score = 0.0
        self.last_threshold = initial_threshold
        
        # Cache for neighbor indices
        self.cached_indices = None
        self.cached_spatial_shape = (0, 0)
        
    def _build_neighbor_indices(
        self, 
        H: int, 
        W: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build grid neighbor indices and relative positions for vectorization.
        
        Returns:
            neighbor_indices: (N, K) LongTensor - indices of K neighbors for each of N nodes
            relative_positions: (K, 2) FloatTensor - relative (dx, dy) for each of the K neighbor types
        """
        N = H * W
        
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        # Flatten grid indices
        node_idx_grid = (ys * W + xs)  # (H, W)
        
        # Generate offsets
        offsets = []
        rel_pos = []
        
        for dy in range(-self.candidate_radius, self.candidate_radius + 1):
            for dx in range(-self.candidate_radius, self.candidate_radius + 1):
                if dy == 0 and dx == 0:
                    continue
                offsets.append((dy, dx))
                rel_pos.append([dx / self.candidate_radius, dy / self.candidate_radius])
                
        # (K, 2)
        relative_positions = torch.tensor(rel_pos, device=device, dtype=torch.float)
        
        # Build neighbor indices for each offset
        neighbor_list = []
        for dy, dx in offsets:
            ny = (ys + dy).clamp(0, H - 1)
            nx = (xs + dx).clamp(0, W - 1)
            neighbor_list.append(node_idx_grid[ny, nx].view(-1))  # (N,)
            
        # Stack to (N, K)
        neighbor_indices = torch.stack(neighbor_list, dim=1)
        
        return neighbor_indices, relative_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) BEV features
            
        Returns:
            Updated features after learned graph message passing
        """
        B, C, H, W = x.shape
        N = H * W
        device = x.device
        
        # 1. Flatten to nodes: (B, N, C)
        nodes = x.permute(0, 2, 3, 1).reshape(B, N, C)
        
        # 2. Get neighbor structure (shared for batch)
        # Check cache
        if self.cached_indices is not None and self.cached_spatial_shape == (H, W):
            neighbor_idx, rel_pos = self.cached_indices
        else:
            # Build and cache
            neighbor_idx, rel_pos = self._build_neighbor_indices(H, W, device)
            # Make sure they are persistent buffers if needed, or just tensors
            self.cached_indices = (neighbor_idx, rel_pos)
            self.cached_spatial_shape = (H, W)
            
        K = neighbor_idx.shape[1]
        
        # 3. Gather neighbor features
        # Expand for batch: (B, N, K)
        batch_neighbor_idx = neighbor_idx.unsqueeze(0).expand(B, -1, -1)
        
        # Gather logic: we need to gather from 'nodes' (B, N, C) using indices (B, N, K)
        # Result should be (B, N, K, C)
        # trick: flatten batch dim for gather, or use simple loop if B is small (usually 2-4)
        # Using loop for clarity and safety with gather:
        h_src = nodes.unsqueeze(2).expand(-1, -1, K, -1)  # (B, N, K, C)
        
        neighbors_flat = batch_neighbor_idx.view(B, N * K)  # (B, N*K)
        # Gather (B, N*K, C)
        # We use gather on dimension 1. nodes is (B, N, C).
        # indices must be (B, N*K, C) -> expand indices to C
        gather_indices = neighbors_flat.unsqueeze(-1).expand(-1, -1, C)
        h_tgt_flat = torch.gather(nodes, 1, gather_indices) # (B, N*K, C)
        h_tgt = h_tgt_flat.view(B, N, K, C)
        
        # 4. Compute Edge Scores
        # Input: cat([h_i, h_j, rel_pos])
        # rel_pos needs expansion: (K, 2) -> (1, 1, K, 2) -> (B, N, K, 2)
        rel_pos_expanded = rel_pos.view(1, 1, K, 2).expand(B, N, -1, -1)
        
        if self.use_position_encoding:
            edge_input = torch.cat([h_src, h_tgt, rel_pos_expanded], dim=-1)
        else:
            edge_input = torch.cat([h_src, h_tgt], dim=-1)
            
        # (B, N, K, 1) -> (B, N, K)
        edge_scores = torch.sigmoid(self.edge_scorer(edge_input).squeeze(-1))
        
        # 5. Dynamic Selection Logic (Vectorized)
        # We need gradients to flow to self.edge_threshold.
        # Hard comparison (edge_scores >= threshold) blocks gradients.
        # Use soft relaxation for the weights: sigmoid((scores - threshold) / temp)
        
        threshold_val = torch.sigmoid(self.edge_threshold)
        
        # Soft mask for gradient flow
        # When training, we want the "decision" to verify feasibility of the threshold.
        # But for exact sparsity we often use hard usage.
        # Compromise: Use hard mask for top-k logic, BUT use soft mask for weighting.
        
        # Calculate soft "probability of keeping" based on threshold
        # If score > threshold, this is close to 1. If score < threshold, close to 0.
        keep_prob = torch.sigmoid((edge_scores - threshold_val) / self.temperature)
        
        # Boolean mask for hard constraints (top-k) - No gradients here
        # (This is fine, we want the threshold to adjust to "allow" more/less edges in the soft sense)
        mask_threshold = edge_scores >= threshold_val
        
        # Counts per node: (B, N)
        counts = mask_threshold.sum(dim=2)
        
        # Conditions
        min_edges = 3
        max_edges = self.max_edges_per_node
        
        # Get top-MAX edges
        _, top_max_indices = torch.topk(edge_scores, k=min(max_edges, K), dim=2)
        mask_top_max = torch.zeros_like(mask_threshold)
        mask_top_max.scatter_(2, top_max_indices, True)
        
        # Get top-MIN edges
        _, top_min_indices = torch.topk(edge_scores, k=min(min_edges, K), dim=2)
        mask_top_min = torch.zeros_like(mask_threshold)
        mask_top_min.scatter_(2, top_min_indices, True)
        
        # Combined logic per node
        use_max = (counts > max_edges).unsqueeze(-1)  # (B, N, 1)
        use_min = (counts < min_edges).unsqueeze(-1)  # (B, N, 1)
        use_thr = (~use_max) & (~use_min)
        
        final_mask = (use_max & mask_top_max) | (use_min & mask_top_min) | (use_thr & mask_threshold)
        
        # 6. Aggregation
        # neighbor_feats: apply MLP to h_tgt
        neighbor_feats = self.message_mlp(h_tgt)  # (B, N, K, C)
        
        # Weighted sum:
        # We multiply by `keep_prob` to allow gradients to flow to threshold!
        # If threshold is too high, keep_prob drops, changing the output --> Loss gradient tells threshold to lower (maybe).
        # We also multiply by final_mask (hard) to ensure sparsity.
        # Note: If mask is 0, gradient is killed for that edge. That's acceptable for pruning.
        # But for edges entering/leaving the set, we rely on the ones near the boundary.
        
        # Improve: weight = edge_scores * keep_prob
        # If score is high and threshold is low -> keep_prob ~ 1 -> weight = score
        # If score is high but threshold is higher -> keep_prob ~ 0 -> weight reduced
        weights = edge_scores * keep_prob 
        
        # Apply Hard Mask for actual sparsity (at inference time this is crucial, during train helps noise)
        weights = weights * final_mask.float()
        
        weights = weights.unsqueeze(-1)  # (B, N, K, 1)
        
        # Weighted messages
        weighted_msgs = neighbor_feats * weights
        
        aggregated = weighted_msgs.sum(dim=2)  # (B, N, C)
        weight_sum = weights.sum(dim=2)  # (B, N, 1)
        
        # Normalize
        aggregated = aggregated / (weight_sum + 1e-6)
        
        # Residual + Project
        output = nodes + self.output_proj(aggregated)
        
        # 7. Statistics
        with torch.no_grad():
            self.last_num_edges = final_mask.sum().item() / B
            self.last_avg_score = (edge_scores * final_mask.float()).sum().item() / max(final_mask.sum().item(), 1)
            self.last_threshold = threshold_val.item()
            
        return output.reshape(B, H, W, C).permute(0, 3, 1, 2)
    
    def get_edge_stats(self) -> dict:
        """Return statistics about learned edges for analysis."""
        return {
            "avg_edges_per_sample": self.last_num_edges,
            "avg_edge_score": self.last_avg_score,
            "learned_threshold": self.last_threshold,
            "edges_per_node": self.last_num_edges / 10000 if self.last_num_edges > 0 else 0,
            "max_edges_per_node": self.max_edges_per_node,
        }


class EdgeSparsityLoss(nn.Module):
    """
    L1 regularization on edge scores to encourage sparsity.
    """
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
        
    def forward(self, edge_scores: torch.Tensor) -> torch.Tensor:
        return self.weight * edge_scores.mean()
