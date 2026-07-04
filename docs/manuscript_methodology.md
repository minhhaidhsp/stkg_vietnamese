# ViSTKG-QA — Đặc tả kỹ thuật tham chiếu (rút gọn từ bản thảo)

> File này KHÔNG phải toàn văn bài báo. Đây là bản tóm tắt các quyết định kỹ
> thuật đã chốt, dùng làm nguồn đối chiếu chung giữa bản thảo và codebase.
> Khi có xung đột giữa code và file này, dừng lại hỏi tác giả — không tự
> suy đoán theo hướng nào đúng.

---

## 1. Kiến trúc mô hình (Mục 3 bản thảo)

### 1.1. Backbone
- **Vintern-1B-v2** (`5CD-AI/Vintern-1B-v2` trên HuggingFace): InternViT-300M
  (vision encoder) + Qwen2-0.5B-Instruct (language model).
- **Hidden size thật của Qwen2-0.5B = 896** (14 heads × 64), lấy từ
  `config.json` thật của model, KHÔNG phải giả định.
- Toàn bộ trọng số gốc của backbone (vision encoder + language model) bị
  **đóng băng hoàn toàn**. Không fine-tune full-weight ở bất kỳ đâu.

### 1.2. Thành phần huấn luyện (trainable)
1. Bảng nhúng trạng thái không gian thời gian: STE (entity), RE (relation).
2. Lớp chiếu `W_q`, `W_v` (đưa E_Q, E_I về không gian chung, CT 5).
3. Lớp chiếu `query_projection` (512 → 896): chiếu `q_m^Z` từ không gian
   nhúng đồ thị (d=512) sang đúng hidden size của backbone (896) trước khi
   dùng trong chú ý chéo CT (7). **Đây là lớp mới phát sinh khi code hóa,
   không giả định 512 trùng khớp chiều ẩn backbone.**
4. Module đánh giá độ tin cậy bộ ba trực quan (học được, KHÔNG dùng công
   thức tuyến tính cố định như `reliability_scorer.py` bản cũ).
5. LoRA adapters: rank r=16, alpha=32, gắn vào ma trận query/value của các
   lớp attention trong LLM.
6. Lớp chiếu `retrieval_query_projection` (896 → 512): CT(6) cần
   `cos(x̄, g(f))`, nhưng `x̄` (gộp trung bình `X`) đã ở llm_hidden_size=896
   (vì `X` qua `W_q/W_v`), còn `g(f)` ở d=512 (không gian STE/RE). Phát
   hiện khi wire `MultimodalEncoder` thật (Bước 4) — lớp MỚI, TÁCH RIÊNG
   với `query_projection` dù cùng cặp chiều (896→512 vs 512→896), vai trò
   khác nhau (chiếu `x̄` VỀ không gian STKG trung tâm, không phải chiếu
   embedding đồ thị LÊN không gian LLM).
7. Lớp chiếu `ranking_projection` (896 → 512, trong `EntityRankingHead`):
   CT(8) cùng lỗi lệch chiều — `u` (từ luồng ẩn cuối LLM, 896) và `STE(e)`
   (512). Xử lý NHẤT QUÁN cùng hướng với mục 6 (chiếu vector truy vấn
   xuống 512, không chiếu ngược `STE(e)` lên 896 — rẻ hơn vì chỉ 1 vector
   mỗi câu hỏi thay vì chiếu lại toàn bộ tập thực thể E). TÁCH RIÊNG với
   `retrieval_query_projection` dù cùng in/out dim — không dùng chung 1
   lớp cho 2 vai trò khác nhau (truy xuất subgraph vs xếp hạng cuối).

### 1.3. Công thức cốt lõi (đánh số theo bản thảo)

**CT (1)** — Sự kiện sáu ngôi:
```
f = (h, r, t, τ, l_h, l_t)
```
h, r, t: thực thể đầu / quan hệ / thực thể đuôi.
τ: thời điểm/khoảng thời gian. l_h, l_t: tọa độ ĐỘC LẬP của hai phía.

**CT (2)** — Điểm hợp lệ hình học (nhúng tịnh tiến):
```
s(f) = || STE(h) + RE(r) - STE(t) ||     (chuẩn L1 hoặc L2)
```

**CT (3)** — Mã hóa câu hỏi: `E_Q = Enc_LM(Q)`, E_Q ∈ R^(n×d)

**CT (4)** — Mã hóa ảnh: `E_I = ViT(I)`, E_I ∈ R^(Np×dv)

**CT (5)** — Hợp nhất đa phương thức:
```
X = [E_Q · W_q ; E_I · W_v]     (nối chuỗi theo trục token)
```

**CT (6)** — Truy xuất có nhận thức không gian thời gian (luồng hướng ra,
ĐÓNG GÓP KỸ THUẬT TRUNG TÂM của bài):
```
rel(f) = α · cos(x̄, g(f)) − (1 − α) · ŝ(f)
```
- x̄: gộp trung bình các hàng của X — X ở llm_hidden_size=896, PHẢI chiếu
  qua `retrieval_query_projection` (896→512, mới, học được — xác nhận khi
  wire thật, xem Mục 1.2 mục 6) về d=512 trước khi tính cosine với g(f).
- g(f): nhúng sự kiện = trung bình STE(h), RE(r), STE(t), đã ở d=512.
- ŝ(f): điểm CT(2) đã chuẩn hóa min-max về [0,1] trên tập ứng viên.
- α ∈ [0,1]: hệ số cân bằng, mặc định 0.5, chọn bằng grid search trên
  validation trong tập {0.1, 0.3, 0.5, 0.7, 0.9}.
- Chọn top-K (K=32) sự kiện cao nhất, KHÔNG dùng ngưỡng cứng θ.
- **Ablation bắt buộc:** α=1 ⇒ tương đương retrieval thuần ngữ nghĩa của
  KG-Attention gốc [21] — đây là bằng chứng trực tiếp cho novelty của CT(6).

**CT (7)** — Chú ý chéo (luồng hướng vào, chuẩn, KHÔNG phải đóng góp mới):
```
r_m = Σ_i softmax((q_m^Z)ᵀ k_i^X / √d) · v_i^X
```
- k_i^X, v_i^X: tái sử dụng ma trận chiếu key/value CÓ SẴN của backbone
  (không phát sinh tham số mới ở bước này).
- q_m^Z: chiếu từ g(f) qua `query_projection` (512→896).
- Đóng góp nằm ở CÁCH xây dựng q_m^Z (từ sự kiện đã sàng lọc CT 6), không
  nằm ở bản thân phép chú ý chéo.
- R_final = tổng có trọng số của {r_m} theo rel(f), cộng dư vào X.
- **Xác nhận thật khi tải Vintern-1B-v2 (Bước 4):** Qwen2-0.5B dùng GQA
  (Grouped Query Attention), KHÔNG phải MHA chuẩn — `k_proj`/`v_proj` chiếu
  896→128 (2 KV-head × 64), không phải 896. `d` trong công thức `√d` ở
  trên là **head_dim=64** (không phải llm_hidden_size=896). q_m^Z (896)
  reshape thành 14 query-head × 64; K/V (128) reshape 2 KV-head × 64 rồi
  `repeat_interleave` theo group_size=7 để khớp 14 query-head (đúng cách
  Qwen2 tự tính attention nội bộ). Nối 14 head lại (896) rồi qua `o_proj`
  THẬT của backbone (tái sử dụng, không tham số mới) để ra r_m ở 896.
  Xem `src/model/fusion.py`.

**CT (8)** — Đầu ra: xếp hạng thực thể (Phương án A, ĐÃ CHỐT — KHÔNG sinh
văn bản tự do / autoregressive generation):
```
p(e | Q, I, G) = softmax(u^T · STE(e))   với e ∈ toàn bộ tập thực thể E
```
Lý do chốt phương án này: tương thích trực tiếp với Hit@K/MRR, nhất quán
với baseline nhúng đồ thị, tránh mâu thuẫn giữa metric xếp hạng và cơ chế
sinh text tự do (lỗi đã phát hiện và sửa ở bản thảo V1).
- u: suy ra từ luồng ẩn cuối LLM (896), PHẢI chiếu qua `ranking_projection`
  (896→512, mới, học được, TÁCH RIÊNG với `retrieval_query_projection`)
  về d=512 trước khi nhân với STE(e) — cùng lỗi lệch chiều như CT(6), xử
  lý nhất quán cùng hướng (xem Mục 1.2 mục 7).

**CT (9)** — Mục tiêu đa nhiệm:
```
L_total = λ1·L_QA + λ2·L_STKG + λ3·L_VG
λ1=1.0, λ2=0.5, λ3=0.3
```
- L_QA: cross-entropy trên phân phối CT(8).
- L_STKG: loss dự đoán liên kết trên s(f) (negative sampling bằng thay
  thế thực thể ngẫu nhiên).
- L_VG: loss module tin cậy bộ ba trực quan.

---

## 2. Cấu hình đã chốt (khớp `config.yaml`)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| N_x, N_y (lưới không gian) | 64, 64 | bbox Việt Nam (8–23.5°N, 102–110°E) |
| N_t (lát thời gian) | 100 | ~12 năm/lát trên range [800, 2025] |
| TAU_MIN, TAU_MAX | 800, 2025 | khớp dữ liệu thật, KHÔNG dùng 2024 |
| d (chiều nhúng STE/RE) | 512 | không gian nhúng đồ thị nội bộ |
| hidden size backbone (Qwen2-0.5B) | 896 | lấy từ config.json thật |
| K (top-K retrieval) | 32 | |
| α (cân bằng CT 6) | 0.5 | grid search {0.1,0.3,0.5,0.7,0.9} |
| LoRA rank / alpha | 16 / 32 | |
| λ1, λ2, λ3 | 1.0, 0.5, 0.3 | |
| Optimizer | AdamW | |
| LR adapter/projection | 2e-4 | |
| LR bảng nhúng đồ thị | 1e-3 | |
| Weight decay | 0.01 | |
| Batch size | 32 | |
| Max epoch / patience | 20 / 3 (early stop theo MRR val) | |
| Quy tắc biên không gian | clip vào ô biên gần nhất, KHÔNG mở rộng lưới toàn cầu | áp dụng cho ~4.7% facts có tọa độ ngoài bbox VN |
| Quy tắc biên thời gian | loại facts τ ngoài [800,2025] khỏi tập huấn luyện STKG, ghi log số lượng | |

---

## 3. Quy trình xây dựng ViSTQAD (Mục 4.1 bản thảo)

### 3.1. Năm bước gốc theo bản thảo
1. **Kế thừa đồ thị tri thức nguồn** — Wikidata (SPARQL, dùng
   `wdt:P31/wdt:P279*` truy vấn phân cấp thay vì liệt kê QID cứng khi có
   thể) + hai crawler Wikipedia bổ sung (xem 3.2).
2. **Bản địa hóa thực thể** — nhãn ưu tiên tiếng Việt, fallback tiếng Anh.
3. **Gán tọa độ địa lý** — GeoNames/OSM cho entity có sẵn P625; bảng tĩnh
   33 tỉnh centroid làm fallback khi OSM Nominatim bị chặn (403) trong môi
   trường sandbox — ĐÃ XÁC NHẬN LÀ GIẢI PHÁP CHÍNH THỨC, không phải tạm bợ.
4. **Đối sánh hình ảnh** — Wikimedia Commons qua Wikidata P18.
5. **Sinh câu hỏi** — từ khuôn mẫu (template) theo ràng buộc không gian
   thời gian của sự kiện sáu ngôi, **sau đó viết lại bằng LLM để đa dạng
   hóa cấu trúc câu, bảo toàn ràng buộc gốc**.
   ⚠️ **CẦN XÁC NHẬN VỚI CODEBASE THỰC TẾ:** nếu pipeline hiện tại CHỈ sinh
   thẳng từ template (không có bước viết lại bằng LLM), phải chọn một
   trong hai: (a) thêm bước viết lại bằng LLM để khớp mô tả bản thảo, hoặc
   (b) sửa lại đoạn mô tả trong bản thảo cho khớp thực tế (không paraphrase
   bằng LLM, chỉ đa dạng qua nhiều template/quan hệ). KHÔNG được để mô tả
   và code lệch nhau.

### 3.2. Nguồn dữ liệu thật đã dùng (khác với dự kiến ban đầu)
Do 5 QID cứng ban đầu cho locatedIn/occurredAt trên Wikidata quá hẹp
(652 facts gốc), đã bổ sung:
- `wikipedia_timeline_collector.py`: parse trang "Niên biểu lịch sử Việt
  Nam" + thể loại "Sự kiện lịch sử Việt Nam" (regex trên cú pháp
  năm/ngày: mô tả) → facts occurredAt.
- `wikipedia_heritage_collector.py`: parse "Danh sách Di tích quốc gia
  Việt Nam" theo từng tỉnh (pandas.read_html trên wikitable) → facts
  locatedIn. τ của nguồn này là **năm công nhận di tích**, không phải năm
  xây dựng gốc — gắn nhãn `temporal_type: recognition_year` để phân biệt
  với τ của occurredAt (năm sự kiện thật).
- `wikidata_collector.py` sửa dùng `wdt:P31/wdt:P279*` với root
  Q1081138 (historic site) — kết quả: đóng góp thực tế gần như 0 (5 QID
  cũ KHÔNG phải lớp con của Q1081138, là nhánh phân loại song song). Giữ
  nguyên truy vấn này vì đúng tinh thần tra cứu phân cấp, nhưng coi
  Wikipedia Heritage crawler là nguồn chính cho locatedIn.

### 3.3. Quy mô dữ liệu thật (THAY THẾ số 10.000/15.420/12.000 trong bản
thảo V1 — bản thảo PHẢI cập nhật theo số này, không phải ngược lại)

| Chỉ số | Giá trị thật |
|---|---|
| Tổng facts (sau dedup) | 6.980 |
| Tổng thực thể duy nhất | 7.218 |
| Câu hỏi sinh ra | 17.450 (⚠️ xác nhận đây là số ổn định cuối trước khi đưa vào bản thảo) |
| bornIn | 4.163 facts (59.7%) |
| diedIn | 848 facts (12.2%) |
| locatedIn | 1.682 facts (24.1%) |
| occurredAt | 287 facts (4.0%) |
| Phân bố câu hỏi (sau cân bằng bằng nhiều template hơn cho nhóm ít) | bornIn 47.7%, locatedIn 36.4%, diedIn 9.7%, occurredAt 6.2% |
| Phủ tọa độ l_t (sau geocode đầy đủ) | 95.7% tổng thể |
| Phủ tọa độ l_h | thấp với locatedIn (~0%, THEO THIẾT KẾ — chỉ geocode cấp tỉnh cho t, h là di tích cụ thể không có tọa độ riêng) |
| Facts loại do τ ngoài [800,2025] | 49 (0.74%), chủ yếu mốc huyền sử TCN |
| Ảnh Wikimedia | CHƯA làm cho ~5.700 thực thể mới (chỉ có ở 652 facts gốc) — việc còn tồn đọng |

### 3.4. Kiểm định chất lượng (bắt buộc, đang tồn đọng)
- 500 mẫu (stratified theo quan hệ) đã xuất ra
  `data/vistqad/validation_sample_500.csv`, CẦN 2 người thẩm định độc lập
  điền cột `annotator1_valid`/`annotator2_valid` (1/0) theo ba tiêu chí:
  đúng ngữ pháp, đúng ngữ nghĩa, bảo toàn ràng buộc không gian thời gian.
  Sau đó chạy `compute_kappa.py`.
- BLEU/ROUGE-L: đo giữa **câu hỏi sau viết lại và khuôn mẫu gốc** (đúng
  mô tả Mục 4.1), KHÔNG phải self-BLEU (đo đa dạng nội bộ giữa các câu
  hỏi với nhau — đây là phép đo khác, đã bị nhầm ở một vòng chạy, cần sửa
  lại nếu pipeline có bước viết lại bằng LLM theo 3.1 bước 5).

---

## 4. Baselines (Mục 4.2 bản thảo, ba nhóm)

1. **Nhúng đồ thị:** EmbedKGQA, CronKGQA, TempoQR, SubGTR — retrain trên
   ViSTQAD bằng mã gốc, thay bộ mã hóa câu hỏi tiếng Anh bằng
   PhoBERT/đa ngữ (ghi rõ cách làm thật).
2. **LLM kết hợp KG:** Think-on-Graph (training-free, agent LLM duyệt đồ
   thị sáu ngôi tuyến tính hóa) + GenTKGQA (retrain).
3. **VLM đa phương thức:** Vintern-1B-v2 (zero-shot + fine-tune),
   Qwen2.5-VL-7B (zero-shot).
- Câu trả lời dạng text tự do (nhóm 2, 3): chuẩn hóa chuỗi (NFC, thường
  hóa, bỏ dấu câu) rồi so khớp chính xác để tính Hit@1; Hit@3/10, MRR chỉ
  áp dụng cho phương pháp có đầu ra xếp hạng.

---

## 5. Ablation bắt buộc (Bảng 6 bản thảo, 8 kịch bản)

1. Mô hình đầy đủ (baseline so sánh).
2. Bỏ luồng chú ý hướng vào.
3. Bỏ định vị không gian.
4. Bỏ nhúng thời gian.
5. **α = 1** (truy xuất thuần ngữ nghĩa, tương đương [21]) — bằng chứng
   novelty của CT(6), KHÔNG được bỏ qua.
6. Bỏ kênh ảnh hoàn toàn — đối chiếu với biến thể "không dùng ảnh" ở
   Bảng 5 để chứng minh cải thiện không chỉ đến từ kênh ảnh.
7. Bỏ trọng số tin cậy bộ ba trực quan.
8. Chỉ huấn luyện L_QA (bỏ đa nhiệm).
9. Thay chú ý hai chiều bằng nối chuỗi đồ thị con vào prompt (kiểu RAG).

Kèm: phân tích theo loại câu hỏi (Bảng 7), đa seed (3–5) + bootstrap
ghép cặp tính p-value (Bảng 5), đo thời gian suy luận (Hình 2), phân loại
lỗi ~100 mẫu (Bảng 8).

---

## 6. Nguyên tắc bất biến xuyên suốt dự án

1. **Không bịa số liệu thực nghiệm.** Mọi ô số liệu trong bản thảo chỉ
   được điền sau khi có kết quả chạy thật. Nếu số thật khác số đang ghi
   trong bản thảo, SỬA BẢN THẢO, không sửa số liệu.
2. **Không tự đóng vai người thẩm định** (Cohen's κ, đánh giá chất lượng
   câu hỏi) — việc này chỉ con người thật làm được.
3. **Mọi trích dẫn học thuật mới phải có nguồn thật, xác minh được** (đã
   có danh mục 41 tài liệu tham khảo hoàn chỉnh, xem file bản thảo).
4. **Khi code và bản thảo lệch nhau, dừng lại hỏi**, không tự chọn một
   bên rồi âm thầm sửa bên kia.
5. **Mô tả kiến trúc = code thật đang chạy.** Không mô tả một cơ chế
   (VD sinh text tự do, hoặc paraphrase bằng LLM) nếu code không thực sự
   làm việc đó.
