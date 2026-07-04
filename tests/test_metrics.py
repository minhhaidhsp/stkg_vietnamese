import torch

from src.eval.metrics import compute_rank, hit_at_k, mrr, summarize_ranks


def test_compute_rank_top1():
    p = torch.tensor([0.1, 0.7, 0.2])
    assert compute_rank(p, 1) == 1


def test_compute_rank_last():
    p = torch.tensor([0.7, 0.2, 0.1])
    assert compute_rank(p, 2) == 3


def test_hit_at_k():
    ranks = [1, 2, 5, 1, 3]
    assert hit_at_k(ranks, 1) == 2 / 5
    assert hit_at_k(ranks, 3) == 4 / 5
    assert hit_at_k(ranks, 10) == 1.0


def test_mrr():
    ranks = [1, 2, 4]
    expected = (1 / 1 + 1 / 2 + 1 / 4) / 3
    assert abs(mrr(ranks) - expected) < 1e-9


def test_summarize_ranks_empty():
    out = summarize_ranks([])
    assert out["mrr"] == 0.0
    assert out["n"] == 0


def test_summarize_ranks_shape():
    out = summarize_ranks([1, 2, 3], ks=(1, 3))
    assert set(out.keys()) == {"hit@1", "hit@3", "mrr", "n"}
