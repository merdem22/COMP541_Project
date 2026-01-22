"""
Graph Module - BEV Reasoning Blocks

Default is conv-based (fast, stable). Optionally supports a lightweight
GNN-style message passing module on downsampled BEV nodes (no extra deps).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class SimpleGraphModule(nn.Module):
    """
    Simple graph-style reasoning using stacked convolutions.

    No attention, no complex graph operations - just convolutions.
    Fast and stable baseline that won't bottleneck training.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.num_layers = num_layers

        # Build conv layers
        layers = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else hidden_channels
            ch_out = hidden_channels if i < num_layers - 1 else out_channels

            layers.append(nn.Conv2d(ch_in, ch_out, kernel_size, padding=kernel_size // 2, bias=False))
            layers.append(nn.BatchNorm2d(ch_out))
            if i < num_layers - 1:  # No ReLU after last layer (before residual)
                layers.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*layers)

        # Residual projection if dimensions don't match
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.out_channels = out_channels

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, C, H, W) BEV features
        Returns:
            out: (B, C, H, W) enhanced BEV features
        """
        out = self.conv_layers(feat)

        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(feat)
        else:
            residual = feat

        return F.relu(out + residual)


class DenseGraphModule(nn.Module):
    """
    Dense Graph Module - uses larger kernels for wider receptive field.
    Still just convolutions, but with larger kernels for "graph-like" context.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 2,
        kernel_size: int = 5,
        num_heads: int = 4,  # Kept for config compatibility, not used
    ):
        super().__init__()

        self.num_layers = num_layers

        # Use larger kernels for wider context (graph-like reasoning)
        layers = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else hidden_channels
            ch_out = hidden_channels

            # Use depthwise separable conv for efficiency with larger kernels
            layers.append(nn.Conv2d(ch_in, ch_in, kernel_size, padding=kernel_size // 2, groups=ch_in, bias=False))
            layers.append(nn.Conv2d(ch_in, ch_out, 1, bias=False))
            layers.append(nn.BatchNorm2d(ch_out))
            layers.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*layers)

        # Output projection
        self.out_proj = nn.Conv2d(hidden_channels, out_channels, 1)

        # Residual projection if needed
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.out_channels = out_channels

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, C, H, W) BEV features
        Returns:
            out: (B, C, H, W) enhanced BEV features
        """
        out = self.conv_layers(feat)
        out = self.out_proj(out)

        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(feat)
        else:
            residual = feat

        return out + residual


class GraphModule(nn.Module):
    """
    Main Graph Module wrapper.

    Simplified: always uses SimpleGraphModule (conv-based).
    The edge_type parameter is kept for config compatibility but ignored.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        k_neighbors: int = 8,  # Kept for config compatibility, not used
        edge_mlp_channels: List[int] = None,  # Kept for config compatibility, not used
        use_edge_features: bool = True,  # Kept for config compatibility, not used
        edge_type: str = 'dense',  # Kept for config compatibility, not used
        stride: int = 4,
        edge_topk: Optional[int] = None,
    ):
        super().__init__()

        edge_type_norm = (edge_type or "").lower()
        if edge_type_norm in {"gnn", "knn", "message_passing"}:
            self.graph = BEVMessagePassingGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=max(1, int(num_layers)),
                k_neighbors=int(k_neighbors),
                stride=int(stride),
                edge_topk=edge_topk,
            )
        else:
            self.graph = SimpleGraphModule(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                kernel_size=kernel_size,
            )

        self.out_channels = out_channels

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, C, H, W) fused BEV features
        Returns:
            out: (B, C, H, W) graph-enhanced BEV features
        """
        return self.graph(feat)


def _make_grid_neighbors(h: int, w: int, k: int, device: torch.device) -> torch.Tensor:
    """
    Build fixed neighborhood indices for an HxW grid.
    Uses offsets in increasing radius; edges are clamped at borders.

    Returns:
        neighbor_idx: (N, k) indices into flattened grid (row-major).
    """
    if k <= 0:
        return torch.empty((h * w, 0), dtype=torch.long, device=device)

    # Generate offsets in a small spiral/ring order: (dy, dx)
    offsets: list[tuple[int, int]] = []
    radius = 1
    while len(offsets) < k:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                if max(abs(dy), abs(dx)) != radius:
                    continue
                offsets.append((dy, dx))
                if len(offsets) >= k:
                    break
            if len(offsets) >= k:
                break
        radius += 1

    offsets = offsets[:k]
    offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)  # (k, 2)

    ys = torch.arange(h, device=device, dtype=torch.long)
    xs = torch.arange(w, device=device, dtype=torch.long)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # (N, 2)

    nbr = coords[:, None, :] + offsets_t[None, :, :]  # (N, k, 2)
    nbr_y = nbr[..., 0].clamp_(0, h - 1)
    nbr_x = nbr[..., 1].clamp_(0, w - 1)
    neighbor_idx = (nbr_y * w + nbr_x).reshape(h * w, k)
    return neighbor_idx


class BEVMessagePassingGNN(nn.Module):
    """
    Lightweight GNN-style message passing on BEV features.

    - Downsample BEV by `stride` to reduce nodes (H' x W').
    - Use a fixed k-neighborhood graph on the grid (no external libs).
    - Do simple message passing (MLP over neighbor deltas) with residuals.
    - Upsample back to original resolution.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
        k_neighbors: int = 8,
        stride: int = 4,
        edge_topk: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.k_neighbors = max(1, int(k_neighbors))
        self.stride = max(1, int(stride))
        self.edge_topk = None if edge_topk is None else max(1, int(edge_topk))

        self.in_proj = None
        if in_channels != out_channels:
            self.in_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # Lightweight edge scoring (attention) projections.
        # Applied per node over its fixed k-neighborhood (after downsampling).
        self.q_proj = nn.Linear(out_channels, out_channels, bias=False)
        self.k_proj = nn.Linear(out_channels, out_channels, bias=False)

        # Per-layer MLPs (delta -> message)
        self.msg_mlps = nn.ModuleList()
        for _ in range(self.num_layers):
            self.msg_mlps.append(
                nn.Sequential(
                    nn.Linear(out_channels, hidden_channels, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_channels, out_channels, bias=False),
                )
            )

        self.norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(self.num_layers)])

        self._cached_hw: Optional[Tuple[int, int]] = None
        self.register_buffer("_neighbor_idx", torch.empty(0, dtype=torch.long), persistent=False)

    def _maybe_build_neighbors(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        if self._cached_hw == (h, w) and self._neighbor_idx.numel() > 0 and self._neighbor_idx.device == device:
            return self._neighbor_idx
        neighbor_idx = _make_grid_neighbors(h, w, self.k_neighbors, device=device)
        self._neighbor_idx = neighbor_idx
        self._cached_hw = (h, w)
        return neighbor_idx

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, C, H, W)
        Returns:
            out: (B, out_channels, H, W)
        """
        B, C, H, W = feat.shape

        residual = self.in_proj(feat) if self.in_proj is not None else feat
        x = residual

        # Downsample for graph processing
        if self.stride > 1:
            x_ds = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        else:
            x_ds = x

        _, C2, H2, W2 = x_ds.shape
        N = H2 * W2
        x_flat = x_ds.permute(0, 2, 3, 1).reshape(B, N, C2)  # (B, N, C)

        neighbor_idx = self._maybe_build_neighbors(H2, W2, device=x_flat.device)  # (N, k)

        for mlp, ln in zip(self.msg_mlps, self.norms):
            neigh = x_flat[:, neighbor_idx]  # (B, N, k, C)

            # Learned edge scoring: attention over neighbors (optionally sparsified by top-k).
            q = self.q_proj(x_flat)  # (B, N, C)
            k = self.k_proj(neigh)   # (B, N, k, C)
            logits = (q[:, :, None, :] * k).sum(dim=-1) / (C2 ** 0.5)  # (B, N, k)

            if self.edge_topk is not None and self.edge_topk < logits.shape[2]:
                topk = self.edge_topk
                topv, topi = logits.topk(topk, dim=2)
                logits = topv
                neigh = neigh.gather(
                    2, topi[:, :, :, None].expand(-1, -1, -1, C2)
                )

            attn = torch.softmax(logits, dim=2)  # (B, N, k_sel)
            delta = neigh - x_flat[:, :, None, :]
            msg = mlp(delta)  # (B, N, k_sel, C)
            agg = (msg * attn[:, :, :, None]).sum(dim=2)  # (B, N, C)

            x_flat = ln((x_flat + agg).reshape(-1, C2)).reshape(B, N, C2)

        x_ds_out = x_flat.reshape(B, H2, W2, C2).permute(0, 3, 1, 2).contiguous()

        # Upsample back
        if self.stride > 1:
            x_up = F.interpolate(x_ds_out, size=(H, W), mode="bilinear", align_corners=False)
        else:
            x_up = x_ds_out

        return F.relu(x_up + residual)
