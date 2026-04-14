import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, SPATIAL_DIR, VIETNAM_LAT, VIETNAM_LON

RAW_CSV      = os.path.join(RAW_DIR, "combined_raw.csv")
ENRICHED_CSV = os.path.join(SPATIAL_DIR, "enriched.csv")

# -------------------------------------------------------------------
raw = pd.read_csv(RAW_CSV)
enriched = pd.read_csv(ENRICHED_CSV)

print("=" * 55)
print("KIỂM TRA BƯỚC 2 - BÁO CÁO SPATIAL ENRICHMENT")
print("=" * 55)

# 1. Tổng số facts
print(f"\n✅ Tổng facts: {len(enriched)}  (raw: {len(raw)})")

# 2. Cải thiện độ phủ tọa độ
print("\n📈 Cải thiện độ phủ tọa độ:")
for col in ["l_h_lat", "l_t_lat"]:
    raw_pct = raw[col].notna().mean() * 100
    enr_pct = enriched[col].notna().mean() * 100
    delta   = enr_pct - raw_pct
    status  = "✅" if enr_pct > 80 else "⚠️"
    print(f"   {status} {col}: {raw_pct:.1f}% → {enr_pct:.1f}%  (+{delta:.1f}%)")

# 3. Phủ tọa độ theo loại quan hệ
print("\n📊 Độ phủ l_h_lat theo relation:")
for rel, grp in enriched.groupby("r"):
    pct = grp["l_h_lat"].notna().mean() * 100
    status = "✅" if pct > 80 else "⚠️"
    print(f"   {status} {rel:12s}: {pct:.1f}%  ({len(grp)} facts)")

# 4. Kiểm tra tọa độ trong bbox Việt Nam
def in_vietnam(lat, lon):
    return (VIETNAM_LAT[0] <= lat <= VIETNAM_LAT[1] and
            VIETNAM_LON[0] <= lon <= VIETNAM_LON[1])

has_h = enriched.dropna(subset=["l_h_lat", "l_h_lon"])
has_t = enriched.dropna(subset=["l_t_lat", "l_t_lon"])

h_ok = has_h.apply(lambda r: in_vietnam(r["l_h_lat"], r["l_h_lon"]), axis=1).mean() * 100
t_ok = has_t.apply(lambda r: in_vietnam(r["l_t_lat"], r["l_t_lon"]), axis=1).mean() * 100

print(f"\n🗺️  Tọa độ trong bbox Việt Nam:")
print(f"   {'✅' if h_ok > 90 else '⚠️'} l_h (head): {h_ok:.1f}%")
print(f"   {'✅' if t_ok > 90 else '⚠️'} l_t (tail): {t_ok:.1f}%")

# 5. Mẫu dữ liệu sau enrichment
print("\n📋 5 facts mẫu (đã có tọa độ đầy đủ):")
sample = enriched.dropna(subset=["l_h_lat", "l_t_lat"]).head()
print(sample[["h_label", "r", "t_label", "tau_start", "l_h_lat", "l_t_lat"]].to_string(index=False))

# 6. Kết luận
h_coverage = enriched["l_h_lat"].notna().mean()
t_coverage = enriched["l_t_lat"].notna().mean()
has_cols    = {"h", "r", "t", "tau_start", "l_h_lat", "l_t_lat"}.issubset(enriched.columns)

ok = h_coverage > 0.8 and t_coverage > 0.9 and has_cols and h_ok > 85 and t_ok > 85

print("\n" + "=" * 55)
if ok:
    print("🎉 BƯỚC 2 HOÀN THÀNH - Sẵn sàng sang Bước 3!")
else:
    reasons = []
    if h_coverage <= 0.8:  reasons.append(f"l_h_lat chỉ {h_coverage*100:.1f}% (<80%)")
    if t_coverage <= 0.9:  reasons.append(f"l_t_lat chỉ {t_coverage*100:.1f}% (<90%)")
    if h_ok <= 85:         reasons.append(f"l_h ngoài VN: {100-h_ok:.1f}%")
    if t_ok <= 85:         reasons.append(f"l_t ngoài VN: {100-t_ok:.1f}%")
    print("⚠️  CẦN XEM LẠI:")
    for r in reasons:
        print(f"   - {r}")
