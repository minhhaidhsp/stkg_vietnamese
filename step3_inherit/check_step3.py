import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STEP3_DIR

MERGED_CSV = os.path.join(STEP3_DIR, "merged.csv")

df = pd.read_csv(MERGED_CSV)

print("=" * 58)
print("KIEM TRA BUOC 3 - KE THUA + VIET HOA")
print("=" * 58)

# 1. Tong quan
total     = len(df)
n_h       = df["h"].nunique()
n_r       = df["r"].nunique()
n_t       = df["t"].nunique()
print(f"\nTong facts    : {total}")
print(f"  {'OK' if total >= 750 else 'WARN'} >= 750 facts: {total}")
print(f"Unique h      : {n_h}")
print(f"Unique r      : {n_r}")
print(f"Unique t      : {n_t}")

# 2. Nhan tieng Viet
h_label_ok = (df["h_label"].notna() & (df["h_label"] != "")).mean() * 100
t_label_ok = (df["t_label"].notna() & (df["t_label"] != "")).mean() * 100
print(f"\nNhan tieng Viet:")
print(f"  {'OK' if h_label_ok >= 80 else 'WARN'} h_label: {h_label_ok:.1f}%")
print(f"  {'OK' if t_label_ok >= 70 else 'WARN'} t_label: {t_label_ok:.1f}%")

# 3. Toa do
lat_ok = df["l_h_lat"].notna().mean() * 100
print(f"\nToa do (l_h_lat): {lat_ok:.1f}%")
print(f"  {'OK' if lat_ok >= 60 else 'WARN'} >= 60%")

# 4. Temporal
tau_ok = df["tau_start"].notna().mean() * 100
print(f"\nTemporal (tau_start): {tau_ok:.1f}%")
print(f"  {'OK' if tau_ok >= 70 else 'WARN'} >= 70%")

# 5. Phan bo quan he
print("\nPhan bo quan he:")
for r, cnt in df["r"].value_counts().head(10).items():
    print(f"  {r:20s}: {cnt:5d} facts")

# 6. Phan bo nguon (neu co)
if "source" in df.columns:
    print("\nPhan bo nguon:")
    for src, cnt in df["source"].value_counts().items():
        print(f"  {src:25s}: {cnt:5d}")

# 7. Ket luan
print("\n" + "=" * 58)
checks = [
    total >= 750,
    h_label_ok >= 80,
    lat_ok >= 60,
    tau_ok >= 70,
]
if all(checks):
    print("BUOC 3 HOAN THANH - San sang sang Buoc 4!")
else:
    print("CAN XEM LAI:")
    if total < 750:     print(f"  - Chi co {total} facts (can >= 750)")
    if h_label_ok < 80: print(f"  - h_label chi dat {h_label_ok:.1f}% (can >= 80%)")
    if lat_ok < 60:     print(f"  - Toa do chi dat {lat_ok:.1f}% (can >= 60%)")
    if tau_ok < 70:     print(f"  - Temporal chi dat {tau_ok:.1f}% (can >= 70%)")
