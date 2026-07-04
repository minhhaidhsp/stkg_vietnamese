import pytest
import torch

from src.model.multimodal_encoder import MultimodalEncoder
from tests.dummy_backbones import DummyTextBackbone, DummyVisionBackbone

TEXT_HIDDEN = 32
VISION_HIDDEN = 24


def _build_encoder(cfg):
    text_backbone = DummyTextBackbone(hidden_dim=TEXT_HIDDEN)
    vision_backbone = DummyVisionBackbone(hidden_dim=VISION_HIDDEN)
    return MultimodalEncoder(cfg, text_backbone, vision_backbone, TEXT_HIDDEN, VISION_HIDDEN)


def test_backbone_base_weights_are_frozen(cfg):
    encoder = _build_encoder(cfg)
    for name, param in encoder.text_backbone.named_parameters():
        if "lora_" in name:
            continue
        assert not param.requires_grad, f"Trọng số gốc backbone '{name}' phải bị đóng băng"
    for name, param in encoder.vision_backbone.named_parameters():
        assert not param.requires_grad, f"Vision backbone '{name}' phải bị đóng băng hoàn toàn"


def test_lora_parameters_are_trainable(cfg):
    encoder = _build_encoder(cfg)
    lora_params = [(n, p) for n, p in encoder.text_backbone.named_parameters() if "lora_" in n]
    assert len(lora_params) > 0, "Không tìm thấy tham số LoRA nào — kiểm tra target_modules"
    for name, param in lora_params:
        assert param.requires_grad, f"Tham số LoRA '{name}' phải huấn luyện được"


def test_lora_rank_and_alpha_from_config(cfg):
    encoder = _build_encoder(cfg)
    peft_config = encoder.text_backbone.peft_config["default"]
    assert peft_config.r == cfg["model"]["lora"]["r"]
    assert peft_config.lora_alpha == cfg["model"]["lora"]["alpha"]


def test_projection_output_dim_matches_llm_hidden_size(cfg):
    encoder = _build_encoder(cfg)
    llm_hidden = cfg["model"]["backbone"]["llm_hidden_size"]
    assert encoder.w_q.out_features == llm_hidden
    assert encoder.w_v.out_features == llm_hidden


def test_encode_and_fuse_shapes(cfg):
    encoder = _build_encoder(cfg)
    batch, n_text_tokens, n_patches = 2, 5, 7
    llm_hidden = cfg["model"]["backbone"]["llm_hidden_size"]

    e_q = encoder.encode_text(torch.randn(batch, n_text_tokens, TEXT_HIDDEN))
    e_i = encoder.encode_image(torch.randn(batch, n_patches, VISION_HIDDEN))
    assert e_q.shape == (batch, n_text_tokens, TEXT_HIDDEN)
    assert e_i.shape == (batch, n_patches, VISION_HIDDEN)

    x = encoder.fuse(e_q, e_i)
    assert x.shape == (batch, n_text_tokens + n_patches, llm_hidden)


def test_constructor_rejects_mismatched_projection_dim():
    bad_cfg = {
        "model": {
            "lora": {"r": 16, "alpha": 32, "target_modules": ["q_proj", "v_proj"]},
            "projections": {"w_q_dim": 768, "w_v_dim": 768},  # SAI: phải = llm_hidden_size
            "backbone": {"llm_hidden_size": 896},
        }
    }
    import pytest
    with pytest.raises(ValueError):
        MultimodalEncoder(
            bad_cfg, DummyTextBackbone(TEXT_HIDDEN), DummyVisionBackbone(VISION_HIDDEN),
            TEXT_HIDDEN, VISION_HIDDEN,
        )


def test_from_pretrained_requires_correct_transformers_version(cfg):
    """from_pretrained() (Bước 4) giờ tải Vintern-1B-v2 THẬT — cần
    transformers==4.42.3 (pin trong requirements.txt) + venv riêng (.venv/,
    Python 3.11), KHÔNG chạy được với transformers 5.x của môi trường test
    Bước 3 mặc định. Test này chỉ xác nhận lỗi xảy ra đúng chỗ dự kiến
    (KeyError từ code tùy biến của model khi version lệch), KHÔNG xác nhận
    load thành công — việc đó thuộc bộ smoke-test riêng (.venv/), không
    chạy trong suite pytest nhanh của Bước 3."""
    import transformers
    if transformers.__version__.startswith("4.42"):
        pytest.skip("Đang chạy trong venv có transformers==4.42.3 đúng — bỏ qua, xem smoke test riêng.")
    with pytest.raises(KeyError):
        MultimodalEncoder.from_pretrained(cfg)


def test_null_image_features_shape_and_batchable_with_fuse(cfg):
    """Cơ chế masking mẫu không ảnh (đã duyệt): null_image_embedding phải
    cùng shape cho mọi batch size, và fuse() với nó phải cho X hợp lệ."""
    encoder = _build_encoder(cfg)
    batch, n_text_tokens = 3, 4
    llm_hidden = cfg["model"]["backbone"]["llm_hidden_size"]

    e_i_null = encoder.null_image_features(batch)
    n_null_patches = cfg["model"].get("null_image_patches", 16)
    assert e_i_null.shape == (batch, n_null_patches, VISION_HIDDEN)

    e_q = encoder.encode_text(torch.randn(batch, n_text_tokens, TEXT_HIDDEN))
    x = encoder.fuse(e_q, e_i_null)
    assert x.shape == (batch, n_text_tokens + n_null_patches, llm_hidden)


def test_null_image_embedding_is_trainable(cfg):
    encoder = _build_encoder(cfg)
    assert encoder.null_image_embedding.requires_grad
