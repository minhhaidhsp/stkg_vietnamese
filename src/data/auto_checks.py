"""
Tiền kiểm tự động cho câu hỏi ViSTQAD — chạy trên toàn bộ
data/vistqad/questions.csv (17.450 câu) và đánh cờ riêng
data/vistqad/validation_sample_500.csv.

6 nhóm kiểm tra (yêu cầu tự hành):
  1. Lỗi mã hóa/ký tự hỏng (mojibake), HTML entity sót, thẻ wiki sót.
  2. Placeholder chưa thay ({...}, chuỗi rỗng).
  3. Answer leakage: đáp án xuất hiện nguyên văn trong câu hỏi.
  4. τ ngoài [800,2025] lọt lưới, hoặc câu hỏi hỏi năm nhưng thiếu τ.
  5. Câu hỏi trùng lặp hoàn toàn (cùng text, khác fact_id).
  6. Tên thực thể bất thường (chỉ số, 1 ký tự, QID thô dạng Q12345).

Output:
  - results/dataset/auto_checks.json (thống kê tổng hợp)
  - data/vistqad/questions_flagged.csv (toàn bộ 17.450 + cột auto_flag/auto_reasons)
  - data/vistqad/validation_sample_500.csv (CẬP NHẬT TẠI CHỖ, thêm auto_flag/auto_reasons)
"""

import json
import logging
import os
import re
import sys
from collections import Counter

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Nhóm 1: mã hóa hỏng / HTML entity / thẻ wiki sót ---
MOJIBAKE_RE = re.compile(r"[�]|Ã[\x80-\xbf]|â\x80|Â[\x80-\xbf]")
HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
WIKI_MARKUP_RE = re.compile(r"\[\[|\]\]|\{\{|\}\}|'''|<ref|</ref>|<br\s*/?>")

# --- Nhóm 2: placeholder chưa thay ---
PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_]+\}")

# --- Nhóm 4: câu hỏi có hỏi năm không ---
ASKS_YEAR_RE = re.compile(r"\bnăm\b", re.IGNORECASE)

# --- Nhóm 6: tên thực thể bất thường ---
QID_RAW_RE = re.compile(r"^Q\d+$")
PURELY_NUMERIC_RE = re.compile(r"^\d+([.,]\d+)?$")


def check_encoding_issues(text: str) -> list[str]:
    reasons = []
    if not isinstance(text, str):
        return ["gia_tri_khong_phai_chuoi"]
    if MOJIBAKE_RE.search(text):
        reasons.append("mojibake")
    if HTML_ENTITY_RE.search(text):
        reasons.append("html_entity_sot")
    if WIKI_MARKUP_RE.search(text):
        reasons.append("the_wiki_sot")
    return reasons


def check_placeholder(text: str) -> list[str]:
    reasons = []
    if not isinstance(text, str) or text.strip() == "":
        reasons.append("chuoi_rong")
        return reasons
    if PLACEHOLDER_RE.search(text):
        reasons.append("placeholder_chua_thay")
    return reasons


def check_answer_leakage(question: str, answer: str) -> list[str]:
    if not isinstance(question, str) or not isinstance(answer, str):
        return []
    answer = answer.strip()
    if len(answer) >= 2 and answer.lower() in question.lower():
        return ["answer_leakage"]
    return []


def check_temporal(row: pd.Series) -> list[str]:
    reasons = []
    tau = row.get("tau_start")
    tau_notna = pd.notna(tau)
    if tau_notna:
        tau_val = float(tau)
        if not (800 <= tau_val <= 2025):
            reasons.append("tau_ngoai_800_2025")
    asks_year = isinstance(row.get("question"), str) and bool(ASKS_YEAR_RE.search(row["question"]))
    if asks_year and not tau_notna:
        reasons.append("hoi_nam_nhung_thieu_tau")
    return reasons


def check_weird_entity(label: str) -> list[str]:
    reasons = []
    if not isinstance(label, str) or not label.strip():
        return reasons
    label = label.strip()
    if QID_RAW_RE.match(label):
        reasons.append("qid_tho_chua_giai_quyet")
    if len(label) == 1:
        reasons.append("ten_thuc_the_1_ky_tu")
    if PURELY_NUMERIC_RE.match(label):
        reasons.append("ten_thuc_the_chi_so")
    return reasons


def run_checks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    all_reasons = []

    # Nhóm 5: trùng lặp câu hỏi (tính trước, theo toàn bộ df truyền vào)
    dup_mask = df.duplicated(subset=["question"], keep=False)

    for idx, row in df.iterrows():
        reasons = []
        reasons += check_encoding_issues(row.get("question", ""))
        reasons += [f"dapan_{r}" for r in check_encoding_issues(row.get("answer_label", row.get("answer", "")))]
        reasons += check_placeholder(row.get("question", ""))
        reasons += check_answer_leakage(row.get("question", ""), row.get("answer_label", row.get("answer", "")))
        reasons += check_temporal(row)
        reasons += [f"dapan_{r}" for r in check_weird_entity(row.get("answer_label", row.get("answer", "")))]
        if dup_mask.loc[idx]:
            reasons.append("cau_hoi_trung_lap")
        all_reasons.append(reasons)

    df["auto_reasons"] = ["; ".join(r) for r in all_reasons]
    df["auto_flag"] = [len(r) > 0 for r in all_reasons]
    return df


def summarize(df: pd.DataFrame) -> dict:
    reason_counter = Counter()
    for reasons_str in df["auto_reasons"]:
        if reasons_str:
            reason_counter.update(reasons_str.split("; "))

    return {
        "n_total": len(df),
        "n_flagged": int(df["auto_flag"].sum()),
        "pct_flagged": round(df["auto_flag"].mean() * 100, 2),
        "reason_counts": dict(reason_counter.most_common()),
    }


def main():
    q_path = os.path.join(ROOT, "data", "vistqad", "questions.csv")
    questions = pd.read_csv(q_path)
    logger.info(f"Đọc {len(questions)} câu hỏi từ {q_path}")

    flagged_all = run_checks(questions)
    out_all_path = os.path.join(ROOT, "data", "vistqad", "questions_flagged.csv")
    flagged_all.to_csv(out_all_path, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu {len(flagged_all)} dòng (có auto_flag) -> {out_all_path}")

    summary_all = summarize(flagged_all)
    logger.info(f"Toàn bộ 17.450: {summary_all['n_flagged']} bị đánh cờ ({summary_all['pct_flagged']}%)")

    # --- validation_sample_500.csv: cập nhật tại chỗ ---
    # File này KHÔNG có cột tau_start/fact_id (chỉ giữ qid, relation,
    # template_id, question, answer_label, annotator*) -> join tạm từ
    # questions.csv theo qid để kiểm tra 4 (τ) và 5 (trùng lặp so với TOÀN
    # BỘ 17.450, không chỉ trong 500 mẫu) có ý nghĩa, rồi bỏ cột join khi lưu.
    sample_path = os.path.join(ROOT, "data", "vistqad", "validation_sample_500.csv")
    # Loại auto_flag/auto_reasons nếu file đã có sẵn từ lần chạy trước (idempotent re-run).
    sample_original_cols = [c for c in pd.read_csv(sample_path, nrows=0).columns
                             if c not in ("auto_flag", "auto_reasons")]
    sample = pd.read_csv(sample_path)
    sample_enriched = sample.merge(
        questions[["qid", "fact_id", "tau_start"]], on="qid", how="left"
    )

    dup_question_texts = set(questions.loc[questions.duplicated(subset=["question"], keep=False), "question"])

    reasons_list = []
    for _, row in sample_enriched.iterrows():
        reasons = []
        reasons += check_encoding_issues(row.get("question", ""))
        reasons += [f"dapan_{r}" for r in check_encoding_issues(row.get("answer_label", ""))]
        reasons += check_placeholder(row.get("question", ""))
        reasons += check_answer_leakage(row.get("question", ""), row.get("answer_label", ""))
        reasons += check_temporal(row)
        reasons += [f"dapan_{r}" for r in check_weird_entity(row.get("answer_label", ""))]
        if row.get("question") in dup_question_texts:
            reasons.append("cau_hoi_trung_lap_trong_toan_bo_17450")
        reasons_list.append(reasons)

    sample["auto_reasons"] = ["; ".join(r) for r in reasons_list]
    sample["auto_flag"] = [len(r) > 0 for r in reasons_list]
    sample_checked = sample[sample_original_cols + ["auto_flag", "auto_reasons"]]
    sample_checked.to_csv(sample_path, index=False, encoding="utf-8-sig")
    logger.info(f"Đã đánh cờ {len(sample_checked)} dòng -> {sample_path} (cập nhật tại chỗ, giữ nguyên cột gốc + auto_flag/auto_reasons)")

    summary_sample = summarize(sample_checked)
    logger.info(f"Mẫu 500: {summary_sample['n_flagged']} bị đánh cờ ({summary_sample['pct_flagged']}%)")

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "full_questions_17450": summary_all,
        "validation_sample_500": summary_sample,
    }
    out_report_path = os.path.join(ROOT, "results", "dataset", "auto_checks.json")
    os.makedirs(os.path.dirname(out_report_path), exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu báo cáo tổng hợp -> {out_report_path}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
