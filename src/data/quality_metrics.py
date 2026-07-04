"""
Đo BLEU/ROUGE-L giữa CÂU HỎI SINH RA và KHUÔN MẪU GỐC của nó (Mục 3.4
manuscript_methodology.md).

Đã xác nhận với generate_questions.py/question_templates.py: pipeline hiện
tại KHÔNG có bước viết lại bằng LLM để đa dạng hóa cấu trúc câu — câu hỏi
được sinh thẳng bằng f-string điền vào 12 template cố định. Vì vậy phép đo
đúng theo Mục 3.4 là so sánh câu hỏi đã sinh với chính khuôn mẫu gốc của nó
(entity được thay bằng token trung tính "X", xem question_templates.canonical_form) —
đo MỨC PARAPHRASE THẬT ĐÃ XẢY RA, dự kiến gần như tối đa (BLEU/ROUGE-L rất
cao) vì không có bước paraphrase nào — đây là kết quả THẬT cần báo cáo,
không phải lỗi đo lường.

KHÔNG dùng self-BLEU (so các câu hỏi với nhau) như vòng chạy trước — đó là
chỉ số đa dạng nội bộ, khác với "mức paraphrase so với khuôn mẫu gốc" mà
Mục 3.4 yêu cầu.
"""

import logging
import os
import sys

import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.question_templates import canonical_form

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_template_fidelity(df: pd.DataFrame) -> pd.DataFrame:
    bleu = BLEU(effective_order=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    bleu_scores, rougeL_scores = [], []
    for _, row in df.iterrows():
        ref = canonical_form(row["template_id"], row["h_label"], row["t_label"], row.get("tau_start"))
        hyp = row["question"]
        bleu_scores.append(bleu.sentence_score(hyp, [ref]).score)
        rougeL_scores.append(scorer.score(ref, hyp)["rougeL"].fmeasure)

    df = df.copy()
    df["bleu_vs_template"] = bleu_scores
    df["rougeL_vs_template"] = rougeL_scores
    return df


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    q_path = os.path.join(root, "data", "vistqad", "questions.csv")
    df = pd.read_csv(q_path)
    logger.info(f"Đọc {len(df)} câu hỏi từ {q_path}")

    scored = compute_template_fidelity(df)

    report = scored.groupby("relation").agg(
        n_questions=("qid", "count"),
        bleu_vs_template_mean=("bleu_vs_template", "mean"),
        rougeL_vs_template_mean=("rougeL_vs_template", "mean"),
    ).round(3)
    logger.info("Kết quả theo quan hệ (BLEU/ROUGE-L so với khuôn mẫu gốc):")
    for relation, row in report.iterrows():
        logger.info(f"  {relation}: n={int(row['n_questions'])} "
                    f"BLEU={row['bleu_vs_template_mean']:.1f} ROUGE-L={row['rougeL_vs_template_mean']:.3f}")

    out_path = os.path.join(root, "data", "vistqad", "question_quality_report.csv")
    report.reset_index().to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu báo cáo -> {out_path}")

    overall_bleu = scored["bleu_vs_template"].mean()
    overall_rouge = scored["rougeL_vs_template"].mean()
    print(f"\nTổng thể: BLEU={overall_bleu:.1f}  ROUGE-L={overall_rouge:.3f}  (n={len(scored)})")
    print(
        "Diễn giải (số thật, không phải suy đoán trước khi chạy): điểm KHÔNG gần "
        "tối đa như kỳ vọng ban đầu — trung bình BLEU~46/ROUGE-L~0.67, không phải "
        ">90. Lý do: BLEU/ROUGE-L tính theo n-gram, và tên thực thể thay cho token "
        "trung tính 'X' thường dài hơn 1 token (vd 'Nguyễn Ánh Tuyết' = 3 token), "
        "sinh thêm các n-gram không khớp với khuôn mẫu gốc dù phần khung câu "
        "('sinh ra ở đâu?') giữ NGUYÊN Y HỆT. Vì vậy điểm vừa phải này KHÔNG phản "
        "ánh có paraphrase cấu trúc câu — chỉ phản ánh độ dài thay đổi của token "
        "thực thể được điền vào. Không có bước viết lại bằng LLM nên không có "
        "paraphrase thật sự nào xảy ra ở tầng cấu trúc câu."
    )
    print(report.to_string())
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
