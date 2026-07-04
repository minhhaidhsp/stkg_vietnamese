"""
Xuất 8 mẫu THẬT từ data/vistqad/questions.csv (17.450 câu hỏi) để minh họa
trong bản thảo, theo tiêu chí:
  - Mỗi quan hệ (bornIn, diedIn, locatedIn, occurredAt) ít nhất 1 mẫu.
  - Ưu tiên mẫu có ảnh (fact gốc nằm trong 652 facts cũ đã Wikimedia
    matching — data/visual/visual_enriched.csv, has_image=True). LƯU Ý:
    chỉ bornIn/diedIn có giao với tập 652 facts cũ; locatedIn/occurredAt
    (nguồn Wikipedia mới) KHÔNG có mẫu nào có ảnh — không thể ưu tiên ảnh
    cho 2 quan hệ này, đây là thực tế dữ liệu, không phải lỗi chọn mẫu.
  - Ít nhất 1 mẫu có khoảng thời gian (tau_start != tau_end).
  - Ít nhất 1 mẫu từ template đảo chiều (occurredAt_t4_reverse).
  - KHÔNG chỉnh sửa nội dung câu hỏi — xuất nguyên trạng.

Output: data/vistqad/showcase_samples.json
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_image_lookup() -> dict[str, str]:
    """QID -> URL Wikimedia Commons (KHÔNG phải file cục bộ — data/visual/images/
    chỉ có .gitkeep trong repo hiện tại, ảnh chưa được lưu vật lý; URL commons
    lấy từ data/visual/url_cache.json là thông tin ảnh thật duy nhất có sẵn)."""
    visual = pd.read_csv(os.path.join(ROOT, "data", "visual", "visual_enriched.csv"))
    with open(os.path.join(ROOT, "data", "visual", "url_cache.json"), encoding="utf-8") as f:
        url_cache = json.load(f)
    has_image_qids = set(visual[visual["has_image"] == True]["h"])
    return {qid: url_cache.get(qid, "") for qid in has_image_qids if url_cache.get(qid)}


def build_candidates() -> pd.DataFrame:
    questions = pd.read_csv(os.path.join(ROOT, "data", "vistqad", "questions.csv"))
    enriched = pd.read_csv(os.path.join(ROOT, "data", "spatial", "enriched.csv"))
    image_lookup = load_image_lookup()

    # Lấy tau_end + l_h/l_t từ enriched.csv (questions.csv không lưu tau_end)
    enriched_cols = enriched[["l_h_lat", "l_h_lon", "l_t_lat", "l_t_lon", "tau_end"]].copy()
    enriched_cols["fact_id"] = enriched.index
    merged = questions.merge(enriched_cols, on="fact_id", suffixes=("", "_enriched"))

    merged["image_url"] = merged["h"].map(image_lookup).fillna("")
    merged["has_image"] = merged["image_url"] != ""
    merged["is_interval"] = (
        merged["tau_end"].notna()
        & merged["tau_start"].notna()
        & (merged["tau_start"] != merged["tau_end"])
    )
    merged["is_reverse_template"] = merged["template_id"] == "occurredAt_t4_reverse"
    return merged


def select_showcase(df: pd.DataFrame, seed: int = 42, n_total: int = 8) -> pd.DataFrame:
    rng_state = seed
    chosen_qids: list[str] = []
    chosen_fact_ids: list = []  # loại trừ theo fact_id để 8 mẫu đến từ 8 fact khác nhau (đa dạng hơn)

    def pick(pool: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        pool = pool[~pool["qid"].isin(chosen_qids) & ~pool["fact_id"].isin(chosen_fact_ids)]
        picked = pool.sample(n=min(n, len(pool)), random_state=rng_state)
        chosen_qids.extend(picked["qid"].tolist())
        chosen_fact_ids.extend(picked["fact_id"].tolist())
        return picked

    parts = []

    # 1) Mỗi quan hệ >=1 mẫu, ưu tiên has_image nếu có
    for relation in ["bornIn", "diedIn", "locatedIn", "occurredAt"]:
        rel_df = df[df["relation"] == relation]
        with_image = rel_df[rel_df["has_image"]]
        pool = with_image if len(with_image) > 0 else rel_df
        parts.append(pick(pool, 1))

    # 2) >=1 mẫu có khoảng thời gian (nếu chưa có trong các mẫu đã chọn)
    already = pd.concat(parts)
    if not already["is_interval"].any():
        interval_pool = df[df["is_interval"]]
        if len(interval_pool) > 0:
            parts.append(pick(interval_pool, 1))

    # 3) >=1 mẫu từ template đảo chiều (nếu chưa có)
    already = pd.concat(parts)
    if not already["is_reverse_template"].any():
        reverse_pool = df[df["is_reverse_template"]]
        if len(reverse_pool) > 0:
            parts.append(pick(reverse_pool, 1))

    # 4) Lấp đầy đủ n_total mẫu, ưu tiên has_image trong phần còn lại
    already = pd.concat(parts)
    remaining_needed = n_total - len(already)
    if remaining_needed > 0:
        remaining_pool = df[~df["qid"].isin(chosen_qids)]
        with_image_remaining = remaining_pool[remaining_pool["has_image"]]
        fill_pool = with_image_remaining if len(with_image_remaining) >= remaining_needed else remaining_pool
        parts.append(pick(fill_pool, remaining_needed))

    result = pd.concat(parts).drop_duplicates(subset="qid")
    return result


def to_showcase_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        records.append({
            "qid": row["qid"],
            "question": row["question"],
            "answer": row["answer_label"],
            "answer_entity_id": row["answer_entity"],
            "template_id": row["template_id"],
            "relation": row["relation"],
            "fact_6tuple": {
                "h": row["h"], "h_label": row["h_label"],
                "r": row["relation"],
                "t": row["t"], "t_label": row["t_label"],
                "tau_start": None if pd.isna(row["tau_start"]) else row["tau_start"],
                "tau_end": None if pd.isna(row["tau_end"]) else row["tau_end"],
                "l_h": [
                    None if pd.isna(row["l_h_lat"]) else row["l_h_lat"],
                    None if pd.isna(row["l_h_lon"]) else row["l_h_lon"],
                ],
                "l_t": [
                    None if pd.isna(row["l_t_lat"]) else row["l_t_lat"],
                    None if pd.isna(row["l_t_lon"]) else row["l_t_lon"],
                ],
            },
            "image_url": row["image_url"] or None,
            "is_interval_sample": bool(row["is_interval"]),
            "is_reverse_template_sample": bool(row["is_reverse_template"]),
        })
    return records


def main():
    df = build_candidates()
    showcase = select_showcase(df)
    records = to_showcase_records(showcase)

    out_path = os.path.join(ROOT, "data", "vistqad", "showcase_samples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Đã xuất {len(records)} mẫu -> {out_path}")
    print("\nKiểm tra tiêu chí:")
    relations_covered = {r["relation"] for r in records}
    print(f"  Quan hệ có mặt: {sorted(relations_covered)} (cần đủ 4)")
    print(f"  Số mẫu có ảnh: {sum(1 for r in records if r['image_url'])}")
    print(f"  Có mẫu khoảng thời gian: {any(r['is_interval_sample'] for r in records)}")
    print(f"  Có mẫu template đảo chiều: {any(r['is_reverse_template_sample'] for r in records)}")
    for r in records:
        print(f"  - [{r['relation']:10s}] {r['template_id']:25s} img={'Y' if r['image_url'] else 'N'}  "
              f"interval={r['is_interval_sample']}  {r['question']}")
    return records


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
