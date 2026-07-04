"""
CT (3)-(5) — Bộ mã hóa đa phương thức: bọc backbone Vintern-1B-v2 (đóng
băng hoàn toàn), gắn LoRA (peft) vào LLM, thêm lớp chiếu W_q/W_v.

X = [E_Q · W_q ; E_I · W_v]   (CT 5, nối theo trục token)
W_q, W_v chiếu về llm_hidden_size=896 (không phải d=512) vì X được cộng dư
trực tiếp vào luồng ẩn còn lại của LLM (xem Hình 1 / Bước 1-2).

THIẾT KẾ DEPENDENCY-INJECTION: constructor nhận text_backbone/vision_backbone
làm tham số thay vì tự tải Vintern-1B-v2 bên trong — vì:
  1. Tải Vintern-1B-v2 thật (~1B tham số, cần internet + vài phút) không
     thể chạy trong unit test "CPU, vài giây" như yêu cầu Bước 3.
  2. API forward pass thật của Vintern (InternVLChatModel, trust_remote_code)
     là kiến trúc tùy biến, cần kiểm chứng trên môi trường GPU thật (Colab,
     Bước 4) trước khi wire chi tiết — KHÔNG đoán mò ở đây.
Unit test (Bước 3) dùng backbone giả lập nhỏ (tests/dummy_backbones.py) để
kiểm tra đúng: đóng băng backbone gốc, LoRA chỉ áp vào q_proj/v_proj và có
thể huấn luyện, W_q/W_v chiếu đúng 896. from_pretrained() (dùng Vintern thật)
để dành cho Bước 4 chạy trên Colab.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class MultimodalEncoder(nn.Module):
    def __init__(
        self,
        cfg: dict,
        text_backbone: nn.Module,
        vision_backbone: nn.Module,
        text_hidden_dim: int,
        vision_hidden_dim: int,
    ):
        super().__init__()
        lora_cfg = cfg["model"]["lora"]
        proj_cfg = cfg["model"]["projections"]
        expected_hidden = cfg["model"]["backbone"]["llm_hidden_size"]

        if proj_cfg["w_q_dim"] != expected_hidden or proj_cfg["w_v_dim"] != expected_hidden:
            raise ValueError(
                f"w_q_dim/w_v_dim ({proj_cfg['w_q_dim']}/{proj_cfg['w_v_dim']}) phải "
                f"khớp llm_hidden_size ({expected_hidden})."
            )

        # Đóng băng TOÀN BỘ backbone gốc trước khi gắn LoRA.
        for p in text_backbone.parameters():
            p.requires_grad = False
        for p in vision_backbone.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=list(lora_cfg["target_modules"]),
        )
        self.text_backbone = get_peft_model(text_backbone, lora_config)
        self.vision_backbone = vision_backbone  # LoRA theo manuscript chỉ gắn vào LLM, không gắn vision encoder

        self.w_q = nn.Linear(text_hidden_dim, proj_cfg["w_q_dim"], bias=False)
        self.w_v = nn.Linear(vision_hidden_dim, proj_cfg["w_v_dim"], bias=False)

        # Cơ chế masking mẫu KHÔNG có ảnh (đã anh duyệt): token placeholder
        # HỌC ĐƯỢC, cùng số lượng "patch" (n_null_patches) như ảnh thật sẽ
        # tạo ra, để X luôn cùng shape trong batch bất kể có ảnh hay không —
        # không cần logic padding/độ dài biến thiên ở CrossAttentionFusion.
        # n_null_patches mặc định nhỏ cho mục đích smoke-test (Bước 4); khi
        # trộn ảnh thật, PHẢI đặt lại đúng bằng số patch thật của InternViT
        # (vd 1024 patch với force_image_size=448, patch=14) trước khi dùng
        # batch có cả ảnh thật lẫn placeholder.
        n_null_patches = cfg["model"].get("null_image_patches", 16)
        self.null_image_embedding = nn.Parameter(torch.randn(n_null_patches, vision_hidden_dim) * 0.02)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        """CT (3): E_Q = Enc_LM(Q)."""
        return self.text_backbone(x)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """CT (4): E_I = ViT(I)."""
        return self.vision_backbone(x)

    def null_image_features(self, batch_size: int) -> torch.Tensor:
        """E_I placeholder cho mẫu không có ảnh (~90.7% dữ liệu) — token học
        được, KHÔNG chạy ViT (tiết kiệm compute), đi qua CÙNG W_v như ảnh
        thật để giữ X đồng nhất shape."""
        return self.null_image_embedding.unsqueeze(0).expand(batch_size, -1, -1)

    def fuse(self, e_q: torch.Tensor, e_i: torch.Tensor) -> torch.Tensor:
        """CT (5): X = [E_Q W_q ; E_I W_v], nối theo trục token (dim=1)."""
        proj_q = self.w_q(e_q)
        proj_v = self.w_v(e_i)
        return torch.cat([proj_q, proj_v], dim=1)

    @classmethod
    def from_pretrained(cls, cfg: dict) -> "MultimodalEncoder":
        """Tải Vintern-1B-v2 THẬT (Bước 4). Xác nhận thật (venv Python 3.11,
        transformers==4.42.3, stub flash_attn rỗng — xem requirements.txt):
          - model.language_model: Qwen2ForCausalLM, hidden=896, GQA
            (14 query-head, 2 KV-head, head_dim=64) — dùng làm text_backbone.
          - model.vision_model: InternVisionModel, hidden=1024 — dùng làm
            vision_backbone.
          - model.language_model.model.embed_tokens: bảng nhúng token, dùng
            cho Enc_LM(Q) (CT 3) — TÁCH RIÊNG khỏi self.text_backbone (đã
            LoRA-wrap) vì Enc_LM chỉ là embedding lookup, "các lớp còn lại
            của LLM" (Hình 1) mới là phần LoRA xử lý sau khi cộng dư X.
        """
        from transformers import AutoModel, AutoTokenizer

        name = cfg["model"]["backbone"]["name"]
        vintern = AutoModel.from_pretrained(name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)

        text_hidden_dim = cfg["model"]["backbone"]["llm_hidden_size"]
        vision_hidden_dim = vintern.vision_model.embeddings.patch_embedding.out_channels

        encoder = cls(
            cfg, vintern.language_model, vintern.vision_model,
            text_hidden_dim, vision_hidden_dim,
        )
        encoder.tokenizer = tokenizer
        encoder.embed_tokens = vintern.language_model.model.embed_tokens
        return encoder
