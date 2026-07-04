# results/ — Ledger tổng hợp kết quả ViSTKG-QA

Quy ước: mỗi file kết quả có `generated_at` (ISO timestamp) trong JSON, và
tên file bao gồm `seed` khi áp dụng. `latest.json` (nếu có) trong mỗi thư
mục con luôn trỏ tới lần chạy gần nhất được coi là chính thức — KHÔNG tự
động ghi đè bằng lần chạy nháp/dry-run.

Commit hash tại thời điểm viết ledger này: `a4956ca` (working tree có thay
đổi chưa commit — xem `git status`, chưa commit theo yêu cầu vì chưa được
anh xác nhận commit).

**Cập nhật 2026-07-04 — đã xử lý 8,2% câu hỏi trùng lặp:** loại 771/17.450
câu hỏi trùng text (giữ bản ghi nguồn ưu tiên Wikidata > Wikipedia timeline
> Wikipedia heritage). Còn lại **16.679 câu hỏi** (bornIn 7.728, locatedIn
6.306, diedIn 1.652, occurredAt 993). File cũ trước khi xử lý đã sao lưu ở
`data/vistqad/_archive/questions_before_dedupe_20260704T003351.csv` và
`results/dataset/_archive/*_20260704T003351.*`. Đồ thị G (`data/spatial/enriched.csv`,
6.980 facts) KHÔNG đổi. `auto_checks.json`/Bảng 3/4/7 đã chạy lại trên dữ
liệu sạch. `train.py` dry-run lại PASS (val_MRR=0.1519, 12 mẫu). Chi tiết
đầy đủ + 5 ví dụ trong `docs/DECISIONS.md`.

## results/manuscript_snapshot.json — Tổng hợp 1-file cho bản thảo (MỚI)

`python -m src.eval.export_manuscript_snapshot` — chạy được BẤT KỲ LÚC NÀO
kể cả khi training đang chạy dở nơi khác, CHỈ ĐỌC checkpoint/kết quả đã
lưu trên đĩa (không tự train/eval). Xác nhận thật lần chạy đầu
(2026-07-04): **241/367 giá trị lá có dữ liệu, 126 null, is_provisional=true**
— null chủ yếu ở `training_status`/`main_results`/`ablation`/`figure_data`
vì **chưa có checkpoint C2/C3/ablation nào** (chưa chạy Colab) và
**0/8 baseline** chạy được — đúng thực tế, không phải lỗi script.
`dataset_stats` (Bảng 3/4/7, auto_checks) đã đầy đủ vì lấy từ dữ liệu có
sẵn. Hình 4/5 CHƯA có script xuất dữ liệu nguồn — ghi rõ trong
`metadata.missing`. Chạy lại script này bất cứ lúc nào trong/sau khi C2/C3
chạy trên Colab để cập nhật snapshot mới nhất.

## results/dataset/ — Bước 2 + tiền kiểm chất lượng (L1-L3, CHẠY THẬT)

| File | Trạng thái | Ghi chú |
|---|---|---|
| `auto_checks.json` | ✅ THẬT | Tiền kiểm 17.450 câu + 500 mẫu, 2 bug thật đã sửa (wiki markup sót) |
| `table3_split_overview.csv`, `table3_split_by_relation.csv`, `table3_split_stats.json` | ✅ THẬT | Bảng 3 — facts/thực thể/câu hỏi theo train/val/test |
| `table4_template_category.csv` | ✅ THẬT | Bảng 4 — 12 template + ghi chú không có "thuần thời gian" |
| `table7_question_type_x_image.csv` | ✅ THẬT | Bảng 7 — câu hỏi theo category x has_image |
| `l1_image_matching.log`, `l1_session_start.txt` | 🔄 ĐANG CHẠY | L1 image matching, đóng khung 24h từ 2026-07-03 21:04:46. Tính đến 2026-07-04: **172 ảnh đã tải** (tăng dần, đã bị ngắt và relaunch nhiều lần — resume đúng mỗi lần nhờ url_cache.json + file ảnh local đã tải, không tải lại). **LƯU Ý MÔI TRƯỜNG**: tiến trình nền KHÔNG sống sót qua ranh giới phiên làm việc trong sandbox này — cần relaunch thủ công (`python -m src.data.run_image_matching_l1`) mỗi khi bắt đầu phiên mới nếu muốn L1 tiếp tục tiến triển. |
| `l1_image_matching_report.json` | ⏳ CHƯA CÓ | Sinh ra khi L1 chạy xong (hết 24h hoặc hết QID để xử lý) |
| `human_validation.json` | ⏳ CHƯA CÓ | Cần anh thẩm định thủ công `validation_sample_500_for_annotators.xlsx` trước (2 sheet, 3 tiêu chí) rồi chạy `src/data/compute_kappa.py` |

## results/eval/ — Bước 6 (Colab GPU, KHÔNG chạy được ở đây)

| File | Trạng thái | Ghi chú |
|---|---|---|
| `ablation_run_log.json` | 🧪 DRY-RUN | Chỉ liệt kê 8 lệnh sẽ chạy (`src/eval/run_ablations.py`), KHÔNG thực thi (`--execute` chưa bật) |
| `test_metrics_seed*.json`, `ablation_results.json`, latency/attention/spatial/scaling | ⏳ CHƯA CÓ | Cần checkpoint thật từ C3 (huấn luyện đầy đủ, 3 seed) trên Colab GPU |

## results/training/ — Bước 4 (Colab GPU, KHÔNG chạy được ở đây)

| File | Trạng thái | Ghi chú |
|---|---|---|
| — | ⏳ CHƯA CÓ | `src/train/train.py` đã viết + test dry-run PASS trên tập nhỏ (12 mẫu, CPU), nhưng huấn luyện đầy đủ (13.965 mẫu train, batch 32, tới 20 epoch, 3 seed) cần GPU thật — KHÔNG chạy ở môi trường này |
| `smoke_test` (không lưu file, chỉ log) | ✅ THẬT (local CPU) | 50/50 mẫu thật, forward+backward qua Vintern-1B-v2 thật thành công, loss giảm 30.70→27.58, 0 NaN. **Chạy trên CPU local, KHÔNG phải Colab GPU như yêu cầu ban đầu** — cần anh (hoặc notebook Colab) xác nhận lại trên GPU thật trước khi coi C1 hoàn tất đúng nghĩa |

## results/baselines/ — Track B (đa số cần GPU/repo ngoài, KHÔNG chạy được ở đây)

| File | Trạng thái | Ghi chú |
|---|---|---|
| `baseline_results.json` | ✅ THẬT (nhưng 0/8 baseline chạy được) | Runner (`src/baselines/run_baselines.py`) chạy thật, báo cáo trung thực: 8/8 baseline SKIPPED (chưa clone repo B2, chưa xác minh API B1, hoặc lỗi môi trường transformers version như Vintern zero-shot) — KHÔNG có số liệu giả |

## results/error_analysis/ — Bước 6 (cần checkpoint thật)

⏳ CHƯA CÓ — `src/eval/run_full_eval.py` đã viết phần xuất top-100 mẫu sai, cần checkpoint C3 để chạy.

---

## Việc tồn đọng cần theo dõi (không phải lỗi, chỉ chưa tới lượt/chưa có điều kiện)

1. **L1** đang chạy nền — theo dõi `results/dataset/l1_image_matching.log`, relaunch nếu tiến trình chết giữa chừng (`python -m src.data.run_image_matching_l1`, tự resume nhờ url_cache.json + file ảnh local đã tải).
0. **Notebook Colab C1→C2→C3** (`notebooks/train_colab.ipynb`) đã hoàn chỉnh: chạy tự động không cần xác nhận giữa các giai đoạn (chỉ dừng thật khi `train.py` tự raise lỗi >5% mẫu NaN/epoch — an toàn mới thêm), mỗi cell in dòng `==RESULT==` để copy dán. C2 tự ghi `results/training/best_alpha.txt`, C3 tự đọc — không cần sửa tay giữa cell.
2. **Thẩm định 500 mẫu thủ công** — anh cần điền `data/vistqad/validation_sample_500_for_annotators.xlsx` (2 sheet) rồi chạy `python -m src.data.compute_kappa`.
3. **C1 xác nhận trên Colab GPU thật** — chạy `notebooks/train_colab.ipynb` cell C1.
4. **C2-C5, B1-B3** — cần Colab GPU thật, code đã chuẩn bị sẵn, chưa chạy.
5. **Kịch bản ablation 9** (RAG nối chuỗi thay chú ý 2 chiều) — chưa cài, cần quyết định kiến trúc riêng.
6. **Đa đáp án cho 1 câu hỏi** (8.2% câu hỏi occurredAt trùng text do 1 sự kiện nhiều địa điểm hợp lệ) — cần anh quyết định hướng xử lý trước khi huấn luyện đầy đủ có ý nghĩa (xem docs/DECISIONS.md).
