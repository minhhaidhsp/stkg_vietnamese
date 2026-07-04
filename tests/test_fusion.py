import pytest
import torch
import torch.nn as nn

from src.model.fusion import CrossAttentionFusion


def test_query_projection_shape(cfg):
    fusion = CrossAttentionFusion(cfg)
    z_m = torch.randn(4, 3, 512)
    q = fusion.project_query(z_m)
    assert q.shape == (4, 3, 896)


def test_query_projection_rejects_wrong_input_dim(cfg):
    fusion = CrossAttentionFusion(cfg)
    z_m_wrong = torch.randn(4, 3, 500)  # sai chiều vào (không phải 512)
    with pytest.raises(ValueError):
        fusion.project_query(z_m_wrong)


def test_kv_dim_matches_gqa_not_hidden_size(cfg):
    """Xác nhận thật từ Qwen2-0.5B (GQA): kv_dim = num_kv_heads*head_dim =
    2*64 = 128, KHÔNG phải llm_hidden_size (896)."""
    fusion = CrossAttentionFusion(cfg)
    assert fusion.kv_dim == 128
    assert fusion.num_heads == 14
    assert fusion.num_kv_heads == 2
    assert fusion.head_dim == 64
    assert fusion.group_size == 7


def test_forward_shape_896_with_real_gqa_kv_dim(cfg):
    fusion = CrossAttentionFusion(cfg)
    batch, n_queries, n_tokens = 2, 5, 20
    z_m = torch.randn(batch, n_queries, 512)
    k_x = torch.randn(batch, n_tokens, fusion.kv_dim)   # 128, không phải 896
    v_x = torch.randn(batch, n_tokens, fusion.kv_dim)
    o_proj = nn.Linear(896, 896)

    r_m = fusion(z_m, k_x, v_x, o_proj)
    assert r_m.shape == (batch, n_queries, 896)


def test_forward_rejects_kv_with_wrong_dim(cfg):
    """K/V phải đúng kv_dim=128 (GQA thật) — 896 (giả định MHA cũ, đã sai) hay
    512 (chiều STKG) đều phải bị từ chối."""
    fusion = CrossAttentionFusion(cfg)
    z_m = torch.randn(2, 3, 512)
    o_proj = nn.Linear(896, 896)
    for wrong_dim in (896, 512):
        k_x_wrong = torch.randn(2, 10, wrong_dim)
        v_x_wrong = torch.randn(2, 10, wrong_dim)
        with pytest.raises(ValueError):
            fusion(z_m, k_x_wrong, v_x_wrong, o_proj)


def test_constructor_rejects_mismatched_out_dim():
    bad_cfg = {
        "model": {
            "query_projection": {"in_dim": 512, "out_dim": 768},  # KHÔNG khớp llm_hidden_size
            "backbone": {
                "llm_hidden_size": 896,
                "llm_num_attention_heads": 14,
                "llm_num_key_value_heads": 2,
                "llm_head_dim": 64,
            },
        }
    }
    with pytest.raises(ValueError):
        CrossAttentionFusion(bad_cfg)


def test_constructor_rejects_inconsistent_head_config():
    bad_cfg = {
        "model": {
            "query_projection": {"in_dim": 512, "out_dim": 896},
            "backbone": {
                "llm_hidden_size": 896,
                "llm_num_attention_heads": 14,
                "llm_num_key_value_heads": 3,   # 14 không chia hết cho 3 -> phải lỗi
                "llm_head_dim": 64,
            },
        }
    }
    with pytest.raises(ValueError):
        CrossAttentionFusion(bad_cfg)


def test_gqa_repeat_interleave_group_mapping_is_correct(cfg):
    """Kiểm tra trực tiếp: query-head thuộc group g phải nhận đúng KV-head g
    (không lệch nhóm) — dựng K/V khác nhau hẳn giữa 2 KV-head để phát hiện
    nếu repeat_interleave dùng sai trục/sai thứ tự."""
    fusion = CrossAttentionFusion(cfg)
    batch, n_q, n_tok = 1, 1, 1  # 1 token duy nhất -> softmax attn luôn =1, out = v của token đó

    # KV-head 0 toàn giá trị 1.0, KV-head 1 toàn giá trị -1.0 (rất khác nhau)
    v_head0 = torch.ones(fusion.head_dim)
    v_head1 = -torch.ones(fusion.head_dim)
    v_x = torch.cat([v_head0, v_head1]).view(1, 1, fusion.kv_dim)  # (batch, n_tok, kv_dim)
    k_x = torch.zeros(batch, n_tok, fusion.kv_dim)  # k không ảnh hưởng vì chỉ có 1 token

    z_m = torch.randn(batch, n_q, 512)
    identity_o = nn.Linear(896, 896, bias=False)
    with torch.no_grad():
        identity_o.weight.copy_(torch.eye(896))

    r_m = fusion(z_m, k_x, v_x, identity_o)
    out_per_head = r_m.view(batch, n_q, fusion.num_heads, fusion.head_dim)

    # 7 head đầu (group 0) phải nhận v_head0 (=1.0), 7 head sau (group 1) phải nhận v_head1 (=-1.0)
    assert torch.allclose(out_per_head[0, 0, 0], v_head0, atol=1e-5)
    assert torch.allclose(out_per_head[0, 0, fusion.group_size - 1], v_head0, atol=1e-5)
    assert torch.allclose(out_per_head[0, 0, fusion.group_size], v_head1, atol=1e-5)
    assert torch.allclose(out_per_head[0, 0, -1], v_head1, atol=1e-5)


def test_o_proj_is_reused_not_new_parameter(cfg):
    """o_proj truyền vào PHẢI được dùng trực tiếp (tái sử dụng), fusion không
    tự tạo o_proj riêng — kiểm tra CrossAttentionFusion không có tham số nào
    tên chứa 'o_proj' hay tương đương chiều 896x896 ngoài query_projection."""
    fusion = CrossAttentionFusion(cfg)
    param_names = [name for name, _ in fusion.named_parameters()]
    assert param_names == ["query_projection.weight"], (
        f"CrossAttentionFusion không được có tham số nào khác ngoài query_projection "
        f"(o_proj phải tái sử dụng từ backbone, không tạo mới) — thấy: {param_names}"
    )
