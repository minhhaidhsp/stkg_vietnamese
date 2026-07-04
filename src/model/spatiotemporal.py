"""
CT (1), (2) — Bảng nhúng không gian thời gian STE/RE + điểm hợp lệ hình học.

STE(entity tại 1 fact cụ thể) được lượng tử hóa vào lưới N_x × N_y (không
gian) và N_t (thời gian), rồi tra bảng nhúng theo trục (tách trục: E_x, E_y,
E_t riêng, cộng lại — giữ số tham số O(N_x+N_y+N_t) thay vì O(N_x·N_y·N_t)).
RE(r) là bảng nhúng quan hệ thông thường theo r trong data.relations.

s(f) = ||STE(h) + RE(r) - STE(t)||_2   (CT 2)

Quy tắc biên (khớp config.yaml):
  - Toạ độ ngoài bbox Việt Nam được CLIP vào ô lưới biên gần nhất.
  - τ ngoài [tau_min, tau_max] được CLIP vào lát thời gian biên gần nhất
    (facts thật ngoài khoảng này đã bị loại ở data pipeline — clip ở đây
    chỉ là an toàn số học, không nhằm đưa dữ liệu ngoài kế hoạch vào).
"""

import torch
import torch.nn as nn


class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        grid = cfg["spatiotemporal_grid"]
        self.n_x = grid["n_x"]
        self.n_y = grid["n_y"]
        self.n_t = grid["n_t"]
        self.d = grid["embedding_dim"]

        bbox = grid["spatial_bbox"]
        self.lat_min, self.lat_max = bbox["lat_min"], bbox["lat_max"]
        self.lon_min, self.lon_max = bbox["lon_min"], bbox["lon_max"]

        trange = grid["temporal_range"]
        self.tau_min, self.tau_max = trange["tau_min"], trange["tau_max"]

        relations = cfg["data"]["relations"]
        self.relation_to_idx = {r: i for i, r in enumerate(relations)}

        self.E_x = nn.Embedding(self.n_x, self.d)
        self.E_y = nn.Embedding(self.n_y, self.d)
        self.E_t = nn.Embedding(self.n_t, self.d)
        self.RE = nn.Embedding(len(relations), self.d)

    # ------------------------------------------------------------------
    # Lượng tử hóa lưới
    # ------------------------------------------------------------------

    def _quantize(self, value: torch.Tensor, vmin: float, vmax: float, n_bins: int) -> torch.Tensor:
        clipped = value.clamp(min=vmin, max=vmax)
        frac = (clipped - vmin) / (vmax - vmin + 1e-12)
        idx = (frac * n_bins).long().clamp(max=n_bins - 1)
        return idx

    def quantize_spatial(self, lat: torch.Tensor, lon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ix = self._quantize(lat, self.lat_min, self.lat_max, self.n_x)
        iy = self._quantize(lon, self.lon_min, self.lon_max, self.n_y)
        return ix, iy

    def quantize_temporal(self, tau: torch.Tensor) -> torch.Tensor:
        return self._quantize(tau, self.tau_min, self.tau_max, self.n_t)

    # ------------------------------------------------------------------
    # STE / RE
    # ------------------------------------------------------------------

    def encode_entity(self, lat: torch.Tensor, lon: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """STE(entity) = E_x[ix] + E_y[iy] + E_t[it]. lat/lon/tau: shape (batch,)."""
        ix, iy = self.quantize_spatial(lat, lon)
        it = self.quantize_temporal(tau)
        return self.E_x(ix) + self.E_y(iy) + self.E_t(it)

    def encode_relation(self, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.RE(relation_ids)

    def relation_ids_from_names(self, names: list[str]) -> torch.Tensor:
        return torch.tensor([self.relation_to_idx[n] for n in names], dtype=torch.long)

    # ------------------------------------------------------------------
    # CT (2): s(f)
    # ------------------------------------------------------------------

    def score(
        self,
        lat_h: torch.Tensor, lon_h: torch.Tensor, tau: torch.Tensor,
        relation_ids: torch.Tensor,
        lat_t: torch.Tensor, lon_t: torch.Tensor,
    ) -> torch.Tensor:
        ste_h = self.encode_entity(lat_h, lon_h, tau)
        re_r = self.encode_relation(relation_ids)
        ste_t = self.encode_entity(lat_t, lon_t, tau)
        return torch.norm(ste_h + re_r - ste_t, p=2, dim=-1)
