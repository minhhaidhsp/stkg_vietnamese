import torch

from src.model.ranking_head import EntityRankingHead


def test_output_is_valid_probability_distribution(cfg):
    head = EntityRankingHead(cfg)
    batch = 4
    num_entities = 30
    d = cfg["spatiotemporal_grid"]["embedding_dim"]
    llm_hidden = cfg["model"]["backbone"]["llm_hidden_size"]

    final_hidden = torch.randn(batch, llm_hidden)
    ste_entities = torch.randn(num_entities, d)

    p = head(final_hidden, ste_entities)
    assert p.shape == (batch, num_entities)
    assert (p >= 0).all()
    assert torch.allclose(p.sum(dim=-1), torch.ones(batch), atol=1e-5)


def test_compute_u_projects_to_d(cfg):
    head = EntityRankingHead(cfg)
    llm_hidden = cfg["model"]["backbone"]["llm_hidden_size"]
    final_hidden = torch.randn(2, llm_hidden)
    u = head.compute_u(final_hidden)
    assert u.shape == (2, cfg["spatiotemporal_grid"]["embedding_dim"])


def test_no_generate_method_exists(cfg):
    """Xác nhận đầu ra KHÔNG có API sinh văn bản tự do (generate/beam search)."""
    head = EntityRankingHead(cfg)
    assert not hasattr(head, "generate")
    assert not hasattr(head, "beam_search")


def test_gradient_flows_to_projection(cfg):
    head = EntityRankingHead(cfg)
    llm_hidden = cfg["model"]["backbone"]["llm_hidden_size"]
    d = cfg["spatiotemporal_grid"]["embedding_dim"]
    final_hidden = torch.randn(2, llm_hidden, requires_grad=True)
    ste_entities = torch.randn(10, d)

    p = head(final_hidden, ste_entities)
    loss = -torch.log(p[:, 0] + 1e-12).mean()
    loss.backward()

    assert head.ranking_projection.weight.grad is not None
    assert torch.any(head.ranking_projection.weight.grad != 0)
