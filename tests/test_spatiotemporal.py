import torch

from src.model.spatiotemporal import SpatioTemporalEmbedding


def test_quantize_within_bins(cfg, mock_facts):
    ste = SpatioTemporalEmbedding(cfg)
    ix, iy = ste.quantize_spatial(mock_facts["lat_h"], mock_facts["lon_h"])
    it = ste.quantize_temporal(mock_facts["tau"])
    assert ix.min() >= 0 and ix.max() < ste.n_x
    assert iy.min() >= 0 and iy.max() < ste.n_y
    assert it.min() >= 0 and it.max() < ste.n_t


def test_quantize_clips_out_of_range_values(cfg):
    ste = SpatioTemporalEmbedding(cfg)
    lat = torch.tensor([-1000.0, 1000.0])
    lon = torch.tensor([-1000.0, 1000.0])
    tau = torch.tensor([-1000.0, 100000.0])
    ix, iy = ste.quantize_spatial(lat, lon)
    it = ste.quantize_temporal(tau)
    assert ix[0].item() == 0 and ix[1].item() == ste.n_x - 1
    assert iy[0].item() == 0 and iy[1].item() == ste.n_y - 1
    assert it[0].item() == 0 and it[1].item() == ste.n_t - 1


def test_encode_entity_shape(cfg, mock_facts):
    ste = SpatioTemporalEmbedding(cfg)
    emb = ste.encode_entity(mock_facts["lat_h"], mock_facts["lon_h"], mock_facts["tau"])
    assert emb.shape == (mock_facts["n"], ste.d)


def test_score_shape_and_nonnegative(cfg, mock_facts):
    ste = SpatioTemporalEmbedding(cfg)
    s = ste.score(
        mock_facts["lat_h"], mock_facts["lon_h"], mock_facts["tau"],
        mock_facts["relation_ids"],
        mock_facts["lat_t"], mock_facts["lon_t"],
    )
    assert s.shape == (mock_facts["n"],)
    assert (s >= 0).all()


def test_score_zero_when_h_equals_t_and_relation_embedding_zero():
    """Nếu RE(r)=0 và h,t cùng vị trí/thời điểm thì STE(h)=STE(t) => s(f)=0."""
    import copy
    cfg_local = {
        "spatiotemporal_grid": {
            "n_x": 4, "n_y": 4, "n_t": 4, "embedding_dim": 8,
            "spatial_bbox": {"lat_min": 0.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 10.0},
            "temporal_range": {"tau_min": 0, "tau_max": 10},
        },
        "data": {"relations": ["r0"]},
    }
    ste = SpatioTemporalEmbedding(cfg_local)
    with torch.no_grad():
        ste.RE.weight.zero_()
    lat = torch.tensor([5.0])
    lon = torch.tensor([5.0])
    tau = torch.tensor([5.0])
    rel = torch.tensor([0])
    s = ste.score(lat, lon, tau, rel, lat, lon)
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-6)


def test_relation_ids_from_names(cfg):
    ste = SpatioTemporalEmbedding(cfg)
    names = cfg["data"]["relations"][:2]
    ids = ste.relation_ids_from_names(names)
    assert ids.tolist() == [0, 1]
