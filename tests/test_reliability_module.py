import pytest
import torch

from src.model.reliability_module import VisualReliabilityModule


def test_output_in_unit_interval(cfg):
    module = VisualReliabilityModule(cfg)
    d = cfg["model"]["visual_reliability_module"]["feature_dim"]
    batch = 6
    s, r, o = torch.randn(batch, d), torch.randn(batch, d), torch.randn(batch, d)
    score = module(s, r, o)
    assert score.shape == (batch,)
    assert (score >= 0).all() and (score <= 1).all()


def test_rejects_wrong_feature_dim(cfg):
    module = VisualReliabilityModule(cfg)
    d = cfg["model"]["visual_reliability_module"]["feature_dim"]
    s, r, o = torch.randn(2, d), torch.randn(2, d + 1), torch.randn(2, d)
    with pytest.raises(ValueError):
        module(s, r, o)


def test_has_trainable_parameters(cfg):
    """Khác với reliability_scorer.py cũ (công thức tuyến tính cố định,
    KHÔNG có tham số), module mới PHẢI có tham số huấn luyện được."""
    module = VisualReliabilityModule(cfg)
    params = list(module.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_gradient_flows_through_mlp(cfg):
    module = VisualReliabilityModule(cfg)
    d = cfg["model"]["visual_reliability_module"]["feature_dim"]
    s = torch.randn(4, d, requires_grad=True)
    r = torch.randn(4, d, requires_grad=True)
    o = torch.randn(4, d, requires_grad=True)

    score = module(s, r, o)
    target = torch.ones(4) * 0.8
    loss = torch.nn.functional.mse_loss(score, target)
    loss.backward()

    for name, p in module.named_parameters():
        assert p.grad is not None, f"Không có gradient cho tham số {name}"
        assert torch.any(p.grad != 0), f"Gradient toàn 0 cho tham số {name}"
