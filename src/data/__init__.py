"""
Pipeline dữ liệu ViSTQAD.

Các collector Wikidata/Wikipedia/GeoNames (Bước 1-3 hiện tại của dự án) vẫn
nằm ở step1_collect/, step2_spatial/, step3_*/ tại gốc repo — đã chạy thật và
tạo ra data/spatial/enriched.csv (6.980 facts). Module này (viết ở Bước 2)
sẽ đọc trực tiếp từ enriched.csv để sinh câu hỏi ViSTQAD, gán nhãn loại câu
hỏi, và chia tập train/val/test, KHÔNG chạy lại việc thu thập dữ liệu thô.
"""
