import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import os
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIVQA_DIR    = os.path.join(BASE_DIR, "data", "vivqa")
OUTPUT_PATH  = os.path.join(VIVQA_DIR, "vi_stkg_multimodal.csv")
TRIPLET_PATH = os.path.join(VIVQA_DIR, "visual_triplets.json")
IMAGE_DIR    = os.path.join(VIVQA_DIR, "images")

df       = pd.read_csv(OUTPUT_PATH)
triplets = json.load(open(TRIPLET_PATH, encoding="utf-8")) if os.path.exists(TRIPLET_PATH) else []
n_images = len([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]) if os.path.exists(IMAGE_DIR) else 0

print("=" * 55)
print("KIEM TRA BUOC 3 - Vi-STKG MULTIMODAL")
print("=" * 55)

print(f"\nTong facts         : {len(df)}")
print(f"So anh da download : {n_images}")
print(f"Visual Triplets    : {len(triplets)}")
has_img = (df.get("image_file", pd.Series([""])) != "").sum()
print(f"Facts co anh       : {has_img}")

print("\nDo phu 6 thanh phan:")
for col in ["h", "r", "t", "tau_start", "l_h_lat", "l_t_lat"]:
    if col in df.columns:
        pct = df[col].notna().mean() * 100
        flag = "OK" if pct > 70 else "WARN"
        print(f"  {flag} {col:12s}: {pct:.1f}%")

print("\nPhan bo theo nguon:")
if "source" in df.columns:
    print(df["source"].value_counts().to_string())

if triplets:
    print("\n3 Visual Triplets mau:")
    for t in triplets[:3]:
        print(f"  [{t.get('image_file','')}]")
        print(f"  -> {t.get('subject','')} | {t.get('predicate','')} | {t.get('object','')}")
        cap = t.get("caption", "")
        print(f"  -> caption: {cap[:60]}{'...' if len(cap)>60 else ''}")
        print()

print("=" * 55)
ok = len(df) > 800 and len(triplets) > 100
if ok:
    print("BUOC 3 HOAN THANH - San sang sang Buoc 4!")
else:
    reasons = []
    if len(df) <= 800:    reasons.append(f"Chi co {len(df)} facts (can > 800)")
    if len(triplets) <= 100: reasons.append(f"Chi co {len(triplets)} triplets (can > 100)")
    print("CAN XEM LAI:")
    for r in reasons:
        print(f"  - {r}")
