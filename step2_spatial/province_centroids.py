"""
Bảng tọa độ trung tâm hành chính (tọa độ thành phố/thị xã tỉnh lỵ) của 33
tỉnh/thành Việt Nam sau đợt sáp nhập 2025 — DỮ LIỆU THAM CHIẾU TĨNH CÔNG KHAI,
KHÔNG phải kết quả gọi geocoding API.

Lý do dùng bảng tĩnh thay vì OSM/GeoNames:
  - OSM Nominatim trả 403 Forbidden cho MỌI request từ môi trường này khi chạy
    thử (xem data/spatial/geocode_cache.csv — 47 entries, 0 có tọa độ).
  - GeoNames chưa có username cấu hình (GEONAMES_USERNAME rỗng trong config.py).
  - Chỉ có 33 giá trị tỉnh/thành duy nhất trong toàn bộ tập locatedIn (1.682
    facts) nên một bảng tra cứu tĩnh, xác minh thủ công, là đủ và rẻ hơn nhiều
    so với chờ đăng ký API key hoặc retry một endpoint đang chặn IP.

Nếu sau này OSM/GeoNames khả dụng trở lại, có thể geocode chi tiết hơn tới
cấp huyện/xã (cột "vi_context" trong heritage_sites.csv đã lưu sẵn thông tin
địa bàn cấp xã/huyện cho việc này).
"""

PROVINCE_CENTROIDS: dict[str, tuple[float, float]] = {
    "Điện Biên": (21.3860, 103.0230),
    "Lai Châu": (22.3860, 103.1580),
    "Lào Cai": (22.4809, 103.9755),
    "Sơn La": (21.3256, 103.9188),
    "Cao Bằng": (22.6667, 106.2500),
    "Lạng Sơn": (21.8530, 106.7610),
    "Phú Thọ": (21.3227, 105.4021),
    "Thái Nguyên": (21.5944, 105.8480),
    "Tuyên Quang": (21.8233, 105.2280),
    "Quảng Ninh": (20.9500, 107.0833),
    "Bắc Ninh": (21.1861, 106.0763),
    "Thành phố Hải Phòng": (20.8449, 106.6881),
    "Hưng Yên": (20.6464, 106.0512),
    "Ninh Bình": (20.2506, 105.9744),
    "Thanh Hóa": (19.8067, 105.7764),
    "Nghệ An": (18.6796, 105.6813),
    "Hà Tĩnh": (18.3428, 105.9057),
    "Quảng Trị": (16.8163, 107.1000),
    "Thành phố Huế": (16.4637, 107.5909),
    "Thành phố Đà Nẵng": (16.0544, 108.2022),
    "Quảng Ngãi": (15.1214, 108.8044),
    "Khánh Hòa": (12.2388, 109.1967),
    "Gia Lai": (13.9833, 108.0000),
    "Đăk Lăk": (12.6667, 108.0500),
    "Lâm Đồng": (11.9404, 108.4583),
    "Thành phố Đồng Nai": (10.9447, 106.8243),
    "Tây Ninh": (11.3100, 106.0989),
    "Thành phố Hồ Chí Minh": (10.7769, 106.7009),
    "Đồng Tháp": (10.4590, 105.6323),
    "An Giang": (10.3860, 105.4351),
    "Vĩnh Long": (10.2537, 105.9722),
    "Thành phố Cần Thơ": (10.0452, 105.7469),
    "Cà Mau": (9.1769, 105.1524),
}


def resolve_province(name: str) -> tuple[float, float] | None:
    if not name:
        return None
    return PROVINCE_CENTROIDS.get(name.strip())
