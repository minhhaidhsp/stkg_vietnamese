"""Fixture dùng chung cho unit test: config thật (config.yaml) + dữ liệu giả
lập ~50 facts (mock_data.n_samples), chạy CPU trong vài giây, không cần GPU
và không cần dữ liệu ViSTQAD thật."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def mock_facts(cfg):
    """Sinh n_samples facts giả lập, tọa độ trong bbox VN, tau trong range,
    quan hệ lấy ngẫu nhiên từ data.relations — mô phỏng schema
    data/spatial/enriched.csv mà không cần đọc file thật."""
    n = cfg["mock_data"]["n_samples"]
    g = torch.Generator().manual_seed(42)
    bbox = cfg["spatiotemporal_grid"]["spatial_bbox"]
    trange = cfg["spatiotemporal_grid"]["temporal_range"]
    relations = cfg["data"]["relations"]

    lat_h = torch.rand(n, generator=g) * (bbox["lat_max"] - bbox["lat_min"]) + bbox["lat_min"]
    lon_h = torch.rand(n, generator=g) * (bbox["lon_max"] - bbox["lon_min"]) + bbox["lon_min"]
    lat_t = torch.rand(n, generator=g) * (bbox["lat_max"] - bbox["lat_min"]) + bbox["lat_min"]
    lon_t = torch.rand(n, generator=g) * (bbox["lon_max"] - bbox["lon_min"]) + bbox["lon_min"]
    tau = torch.rand(n, generator=g) * (trange["tau_max"] - trange["tau_min"]) + trange["tau_min"]
    rel_idx = torch.randint(0, len(relations), (n,), generator=g)
    rel_names = [relations[i] for i in rel_idx.tolist()]

    return {
        "n": n,
        "lat_h": lat_h, "lon_h": lon_h,
        "lat_t": lat_t, "lon_t": lon_t,
        "tau": tau,
        "relation_ids": rel_idx,
        "relation_names": rel_names,
    }
