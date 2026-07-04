"""Hit@K và MRR — dùng cho early stopping (Bước 4) và báo cáo Bảng 5 (Bước 6).

Đầu vào là RANK (vị trí, 1-indexed) của đáp án đúng trong danh sách xếp
hạng theo p(e|Q,I,G) giảm dần — không phải điểm số thô, để tách biệt việc
tính hạng (do EntityRankingHead) khỏi việc tổng hợp chỉ số.
"""

import torch


def compute_rank(p: torch.Tensor, target_idx: int) -> int:
    """p: (num_entities,) phân phối xác suất. Trả về rank 1-indexed của target_idx
    (rank=1 nghĩa là target có xác suất cao nhất)."""
    order = torch.argsort(p, descending=True)
    rank = (order == target_idx).nonzero(as_tuple=True)[0].item() + 1
    return rank


def hit_at_k(ranks: list[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= k) / len(ranks)


def mrr(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks) / len(ranks)


def summarize_ranks(ranks: list[int], ks: tuple[int, ...] = (1, 3, 10)) -> dict:
    return {
        **{f"hit@{k}": round(hit_at_k(ranks, k), 4) for k in ks},
        "mrr": round(mrr(ranks), 4),
        "n": len(ranks),
    }
