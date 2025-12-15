"""
Placeholder for future graph reasoning module.

Intended to sit after fusion and before the detection head.
"""

import torch
from torch import nn
import torch.nn.functional as F


class StaticGraphModule(nn.Module):
    """
    Static grid-graph GNN on BEV features.

    - Input:  (B, C, H, W): 
    B = batch size (number of frames/scenes passed in at once), 
    C = number of channels (features per cell), 
    H = height (number of rows in the BEV), 
    W = width (number of columns in the BEV).
    Example: (2, 16, 100, 100) means 2 frames, 16 channels, 100x100 BEV grid.
    - Nodes:  N = H * W (each BEV cell): N = number of cells in the BEV, each cell is a node in the graph.
    - Edges:  fixed 8-connected neighbors in the grid
    - Output: (B, C, H, W): B = batch size, C = number of channels, H = height, W = width.

    This is basically a GraphSAGE layer on a regular lattice,
    implemented without building an NxN adjacency matrix.

    hi′​=ReLU(W⋅[hi​∣∣meanj∈N(i)​hj​]) <- typical GraphSAGE update rule.
    """

    def __init__(self, in_channels: int, k_neighbors: int = 8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.k_neighbors = k_neighbors
        self.proj = nn.Linear(2 * in_channels, in_channels)
        self.activation = nn.ReLU(inplace=True)


    @staticmethod
    def _build_grid_neighbors( H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Build a grid of node indices for a 2D grid.
        """
        
        # ys and xs are 2D tensors giving the y and x coordinates for each grid location:
        ys, xs = torch.meshgrid(torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing="ij") # ys: (H, W), xs: (H, W)
        
        # This flattens 2D (y, x) into a 1D index so each BEV cell has a unique id.
        node_idx = ys * W + xs  # (H, W) → node id


        # neighbour ofsset
        offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        neighbors = []
        
        # ny = ys + dy, nx = xs + dx: used to shift the grid left, right, up, down.
        for dy, dx in offsets:
            # clamp is used to ensure the shifted grid stays within the bounds of the original grid.
            ny = (ys + dy).clamp(0, H - 1) 
            nx = (xs + dx).clamp(0, W - 1)
            neighbors.append(node_idx[ny, nx].reshape(-1))  # (N,)

        neighbor_idx = torch.stack(neighbors, dim=1)
        return neighbor_idx  # long tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """

        B, C, H, W = x.shape
        N = H * W
        device = x.device

        # 1) Flatten grid to nodes: (B, C, H, W) -> (B, N, C)
        x_nodes = x.permute(0, 2, 3, 1).reshape(B, N, C)

        # 2) Get neighbor indices (N, K) for this grid size
        neighbor_idx = self._build_grid_neighbors(H, W, device)  # (N, K)
        K = neighbor_idx.shape[1]

        # 3) Gather neighbor features
        # neighbor_idx_expanded: (B, N, K)
        neighbor_idx_expanded = neighbor_idx.unsqueeze(0).expand(B, -1, -1)

        # Build indices for gather: (B, N, K, C)
        gather_idx = neighbor_idx_expanded.unsqueeze(-1).expand(-1, -1, -1, C)
        # x_nodes_expanded: (B, N, 1, C) broadcast over K for gather
        x_nodes_expanded = x_nodes.unsqueeze(2).expand(-1, -1, K, -1)
        # Gather neighbors: (B, N, K, C)
        x_neighbors = torch.gather(x_nodes_expanded, dim=1, index=gather_idx)

        # 4) Aggregate neighbors (mean over K)
        neighbor_mean = x_neighbors.mean(dim=2)  # (B, N, C)

        # 5) GraphSAGE update: concat(self, neighbor_mean) -> proj -> ReLU
        agg = torch.cat([x_nodes, neighbor_mean], dim=-1)  # (B, N, 2C)
        x_out = self.proj(agg)
        x_out = self.activation(x_out)

        # Optional residual
        x_out = x_out + x_nodes

        # 6) Unflatten back to grid: (B, N, C) -> (B, C, H, W)
        x_out = x_out.permute(0, 2, 1).reshape(B, C, H, W)
        return x_out
