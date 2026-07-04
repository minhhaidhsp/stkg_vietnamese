import torch
import torch.nn.functional as F

from src.model.retriever import SubgraphRetriever

LLM_HIDDEN = 896
D = 512


def _minimal_cfg(alpha=0.5, top_k=32, in_dim=LLM_HIDDEN, out_dim=D):
    return {
        "retrieval": {"alpha": alpha, "top_k": top_k},
        "model": {
            "retrieval_query_projection": {"in_dim": in_dim, "out_dim": out_dim},
            "backbone": {"llm_hidden_size": in_dim},
        },
        "spatiotemporal_grid": {"embedding_dim": out_dim},
    }


def _random_candidates(n, d_in=LLM_HIDDEN, d_out=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    x_bar_raw = torch.randn(d_in, generator=g)     # X chưa chiếu, 896
    g_f = torch.randn(n, d_out, generator=g)         # g(f), đã ở 512
    s_f = torch.rand(n, generator=g) * 10
    return x_bar_raw, g_f, s_f


def test_relevance_shape(cfg):
    retriever = SubgraphRetriever(cfg)
    x_bar_raw, g_f, s_f = _random_candidates(20, LLM_HIDDEN, cfg["spatiotemporal_grid"]["embedding_dim"])
    rel = retriever.relevance(x_bar_raw, g_f, s_f)
    assert rel.shape == (20,)


def test_project_x_bar_shape(cfg):
    retriever = SubgraphRetriever(cfg)
    x_bar_raw = torch.randn(4, LLM_HIDDEN)
    x_bar = retriever.project_x_bar(x_bar_raw)
    assert x_bar.shape == (4, cfg["spatiotemporal_grid"]["embedding_dim"])


def test_project_x_bar_rejects_wrong_input_dim(cfg):
    import pytest
    retriever = SubgraphRetriever(cfg)
    with pytest.raises(ValueError):
        retriever.project_x_bar(torch.randn(4, 512))  # 512 (đã chiếu/g_f) thay vì 896 (X thô)


def test_constructor_rejects_mismatched_projection_dims():
    import pytest
    bad_cfg = _minimal_cfg()
    bad_cfg["model"]["backbone"]["llm_hidden_size"] = 768  # lệch với in_dim=896
    with pytest.raises(ValueError):
        SubgraphRetriever(bad_cfg)


def test_top_k_returns_configured_k(cfg):
    retriever = SubgraphRetriever(cfg)
    x_bar_raw, g_f, s_f = _random_candidates(50, LLM_HIDDEN, cfg["spatiotemporal_grid"]["embedding_dim"])
    rel = retriever.relevance(x_bar_raw, g_f, s_f)
    idx = retriever.top_k_indices(rel)
    assert idx.shape[0] == cfg["retrieval"]["top_k"]


def test_top_k_smaller_than_candidates_is_capped():
    retriever = SubgraphRetriever(_minimal_cfg(top_k=32))
    rel = torch.randn(5)
    idx = retriever.top_k_indices(rel)
    assert idx.shape[0] == 5  # top_k=32 > 5 ứng viên -> chỉ lấy hết 5


def test_alpha_equals_1_matches_pure_cosine_ranking_in_projected_space():
    """Ablation quan trọng nhất: alpha=1 => rel(f) phải cho THỨ HẠNG giống
    hệt cosine similarity thuần túy TRONG KHÔNG GIAN ĐÃ CHIẾU (project_x_bar
    là bước bắt buộc trước cosine, không phải một phần trọng số alpha), bất
    kể s(f) là gì (vì hệ số (1-alpha)=0)."""
    retriever = SubgraphRetriever(_minimal_cfg(alpha=1.0, top_k=10, in_dim=32, out_dim=16))
    g = torch.Generator().manual_seed(1)
    x_bar_raw = torch.randn(32, generator=g)
    g_f = torch.randn(30, 16, generator=g)
    s_f = torch.rand(30, generator=g) * 1000  # s(f) cố ý random lớn, không được ảnh hưởng

    rel = retriever.relevance(x_bar_raw, g_f, s_f, alpha=1.0)

    x_bar_projected = retriever.project_x_bar(x_bar_raw)
    pure_cos = F.cosine_similarity(x_bar_projected.unsqueeze(0).expand_as(g_f), g_f, dim=-1)

    assert torch.allclose(rel, pure_cos, atol=1e-5)

    rank_rel = torch.argsort(rel, descending=True)
    rank_cos = torch.argsort(pure_cos, descending=True)
    assert torch.equal(rank_rel, rank_cos)


def test_alpha_equals_0_is_pure_negative_s_hat():
    retriever = SubgraphRetriever(_minimal_cfg(alpha=0.0, top_k=10, in_dim=32, out_dim=8))
    x_bar_raw, g_f, s_f = _random_candidates(15, 32, 8, seed=2)
    rel = retriever.relevance(x_bar_raw, g_f, s_f, alpha=0.0)
    s_hat = retriever._min_max_normalize(s_f)
    assert torch.allclose(rel, -s_hat, atol=1e-6)


def test_batched_relevance():
    retriever = SubgraphRetriever(_minimal_cfg(alpha=0.5, top_k=5, in_dim=32, out_dim=8))
    batch, n = 4, 12
    x_bar_raw = torch.randn(batch, 32)
    g_f = torch.randn(batch, n, 8)
    s_f = torch.rand(batch, n)
    rel = retriever.relevance(x_bar_raw, g_f, s_f)
    assert rel.shape == (batch, n)


def test_retrieval_query_projection_is_trainable(cfg):
    retriever = SubgraphRetriever(cfg)
    assert retriever.retrieval_query_projection.weight.requires_grad


def test_gradient_flows_to_retrieval_query_projection(cfg):
    retriever = SubgraphRetriever(cfg)
    d = cfg["spatiotemporal_grid"]["embedding_dim"]
    x_bar_raw = torch.randn(3, LLM_HIDDEN, requires_grad=True)
    g_f = torch.randn(3, 5, d)
    s_f = torch.rand(3, 5)
    rel = retriever.relevance(x_bar_raw, g_f, s_f)
    rel.sum().backward()
    assert retriever.retrieval_query_projection.weight.grad is not None
    assert torch.any(retriever.retrieval_query_projection.weight.grad != 0)
