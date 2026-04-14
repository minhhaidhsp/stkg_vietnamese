import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VISUAL_DIR

ENRICHED_CSV  = os.path.join(VISUAL_DIR, "visual_enriched.csv")
TRIPLETS_JSON = os.path.join(VISUAL_DIR, "visual_triplets.json")
FEATURES_NPZ  = os.path.join(VISUAL_DIR, "vit_features.npz")

df = pd.read_csv(ENRICHED_CSV)

print("=" * 58)
print("KIỂM TRA BƯỚC 3 - BÁO CÁO VISUAL ENRICHMENT")
print("=" * 58)

# 1. Tổng quan
total      = len(df)
n_entities = df["h"].nunique()
n_images   = int(df["has_image"].sum())
img_pct    = n_images / n_entities * 100 if n_entities else 0

print(f"\n✅ Tổng facts:    {total}")
print(f"   Unique entities: {n_entities}")
print(f"   {'✅' if img_pct > 30 else '⚠️'} Có ảnh: {n_images} entities ({img_pct:.1f}%)")

# 2. Reliability score
rs = df["reliability_score"]
pct_above = (rs >= 0.5).mean() * 100
print(f"\n📊 Reliability Score:")
print(f"   Mean  : {rs.mean():.3f}")
print(f"   Median: {rs.median():.3f}")
print(f"   Min   : {rs.min():.3f}  |  Max: {rs.max():.3f}")
print(f"   {'✅' if pct_above > 60 else '⚠️'} score ≥ 0.5: {pct_above:.1f}%")

# 3. Per-relation breakdown
print("\n📈 Reliability theo relation:")
for rel, grp in df.groupby("r"):
    m   = grp["reliability_score"].mean()
    img = grp["has_image"].mean() * 100
    print(f"   {rel:12s}: reliability={m:.3f}  |  có ảnh={img:.1f}%  ({len(grp)} facts)")

# 4. Component scores
print("\n🔍 Component scores (trung bình):")
for col in ["vis_score", "clip_score", "spatial_score", "temporal_score"]:
    if col in df.columns:
        print(f"   {col:16s}: {df[col].mean():.3f}")

# 5. Visual triplets
if os.path.exists(TRIPLETS_JSON):
    with open(TRIPLETS_JSON, encoding="utf-8") as f:
        trips = json.load(f)
    n_with_trips  = sum(1 for v in trips.values() if v)
    total_trips   = sum(len(v) for v in trips.values())
    print(f"\n🖼️  Visual Triplets:")
    print(f"   Entities với triplets : {n_with_trips}")
    print(f"   Tổng triplets          : {total_trips}")
    if trips:
        sample_qid = next(q for q, v in trips.items() if v)
        print(f"   Mẫu ({sample_qid}):")
        for t in trips[sample_qid][:2]:
            print(f"     {t}")
else:
    print("\n⚠️  visual_triplets.json chưa có")

# 6. ViT features
if os.path.exists(FEATURES_NPZ):
    import numpy as np
    feats = np.load(FEATURES_NPZ, allow_pickle=False)
    print(f"\n🧠 ViT Features: {len(feats.files)} vectors, dim={feats[feats.files[0]].shape[0] if feats.files else 'N/A'}")
else:
    print("\n⚠️  vit_features.npz chưa có")

# 7. Top 5 facts
print("\n📋 Top 5 facts (reliability cao nhất):")
cols = ["h_label", "r", "t_label", "tau_start", "reliability_score"]
print(df.nlargest(5, "reliability_score")[cols].to_string(index=False))

# 8. Kết luận
has_scores   = "reliability_score" in df.columns
score_ok     = rs.mean() > 0.35
has_triplets = os.path.exists(TRIPLETS_JSON)

print("\n" + "=" * 58)
if has_scores and score_ok and has_triplets:
    print("🎉 BƯỚC 3 HOÀN THÀNH - Sẵn sàng sang Bước 4!")
else:
    reasons = []
    if not has_scores:   reasons.append("Thiếu cột reliability_score")
    if not score_ok:     reasons.append(f"Reliability mean quá thấp ({rs.mean():.3f} < 0.35)")
    if not has_triplets: reasons.append("Thiếu visual_triplets.json")
    print("⚠️  CẦN XEM LẠI:")
    for r in reasons:
        print(f"   - {r}")
