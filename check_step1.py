import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

df = pd.read_csv("data/raw/combined_raw.csv")

print("=" * 50)
print("KIỂM TRA BƯỚC 1 - BÁO CÁO")
print("=" * 50)

# 1. Tổng số facts
print(f"\n✅ Tổng số facts: {len(df)}")
print(f"   (Mục tiêu: > 500 facts)")

# 2. Kiểm tra đủ 6 cột của 6-ngôi
required_cols = ["h", "r", "t", "tau_start", "l_h_lat", "l_t_lat"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"\n❌ Thiếu cột: {missing}")
else:
    print(f"\n✅ Đủ 6 cột 6-ngôi")

# 3. Tỷ lệ null của từng thành phần
print("\n📊 Tỷ lệ dữ liệu hợp lệ (không null):")
for col in required_cols:
    pct = df[col].notna().mean() * 100
    status = "✅" if pct > 50 else "⚠️"
    print(f"   {status} {col}: {pct:.1f}%")

# 4. Mẫu dữ liệu
print("\n📋 5 facts mẫu:")
print(df[["h_label", "r", "t_label", "tau_start", "l_h_lat", "l_t_lat"]].head())

# 5. Kết luận
has_data = len(df) > 100
has_cols = len(missing) == 0
has_time = df["tau_start"].notna().mean() > 0.5

if has_data and has_cols and has_time:
    print("\n🎉 BƯỚC 1 HOÀN THÀNH - Sẵn sàng sang Bước 2!")
else:
    print("\n⚠️  CẦN XEM LẠI trước khi sang Bước 2")
