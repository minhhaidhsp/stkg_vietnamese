"""
Chuyển đổi giữa lịch dương và lịch âm (âm lịch Việt Nam).
Thuật toán dựa trên công trình của Hồ Ngọc Đức (Ho Ngoc Duc).
Tham khảo: https://www.informatik.uni-leipzig.de/~duc/amlich/
"""

import csv
import math
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR


def _jd_from_date(day: int, month: int, year: int) -> int:
    """Tính Julian Day Number từ ngày dương lịch."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn


def _date_from_jd(jd: int) -> tuple[int, int, int]:
    """Tính ngày dương lịch từ Julian Day Number. Trả về (ngày, tháng, năm)."""
    z = jd
    a = z + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + m // 10
    return day, month, year


def _new_moon(k: int) -> float:
    """Tính thời điểm trăng mới thứ k (tính theo Julian Day)."""
    t = k / 1236.85
    t2 = t * t
    t3 = t2 * t
    dr = math.pi / 180.0
    jd1 = 2415020.75933 + 29.53058868 * k + 0.0001178 * t2 - 0.000000155 * t3
    jd1 += 0.00033 * math.sin((166.56 + 132.87 * t - 0.009173 * t2) * dr)
    m = 359.2242 + 29.10535608 * k - 0.0000333 * t2 - 0.00000347 * t3
    mpr = 306.0253 + 385.81691806 * k + 0.0107306 * t2 + 0.00001236 * t3
    f = 21.2964 + 390.67050646 * k - 0.0016528 * t2 - 0.00000239 * t3
    c1 = (0.1734 - 0.000393 * t) * math.sin(m * dr) + 0.0021 * math.sin(2 * dr * m)
    c1 -= 0.4068 * math.sin(mpr * dr) + 0.0161 * math.sin(dr * 2 * mpr)
    c1 -= 0.0004 * math.sin(dr * 3 * mpr)
    c1 += 0.0104 * math.sin(dr * 2 * f) - 0.0051 * math.sin(dr * (m + mpr))
    c1 -= 0.0074 * math.sin(dr * (m - mpr)) + 0.0004 * math.sin(dr * (2 * f + m))
    c1 -= 0.0004 * math.sin(dr * (2 * f - m)) - 0.0006 * math.sin(dr * (2 * f + mpr))
    c1 += 0.0010 * math.sin(dr * (2 * f - mpr)) + 0.0005 * math.sin(dr * (m + 2 * mpr))
    delta = 0.0
    if t < -11:
        delta = 0.001 + 0.000839 * t + 0.0002261 * t2 - 0.00000845 * t3 - 0.000000081 * t * t3
    else:
        delta = -0.000278 + 0.000265 * t + 0.000262 * t2
    return jd1 + c1 - delta


def _sun_longitude(jdn: float) -> float:
    """Tính kinh độ mặt trời tại thời điểm Julian Day (đơn vị: radian/2pi, tức phần của vòng tròn)."""
    t = (jdn - 2451545.0) / 36525.0
    t2 = t * t
    dr = math.pi / 180.0
    m = 357.5291 + 35999.0503 * t - 0.0001559 * t2 - 0.00000048 * t * t2
    lon0 = 280.46646 + 36000.76983 * t + 0.0003032 * t2
    dl = 1.9146 - 0.004817 * t - 0.000014 * t2
    lon = lon0 + dl * math.sin(dr * m) + 0.019993 * math.sin(dr * 2 * m)
    lon = lon - 0.00569 - 0.00478 * math.sin(dr * (125.04 - 1934.136 * t))
    lon = lon * dr
    lon = lon - math.pi * 2 * math.floor(lon / (math.pi * 2))
    return lon


def _get_new_moon_day(k: int, time_zone: float) -> int:
    """Trả về ngày dương lịch của trăng mới thứ k."""
    return int(_new_moon(k) + 0.5 + time_zone / 24)


def _get_lunar_month_11(year: int, time_zone: float) -> int:
    """Tìm tháng 11 âm lịch (tháng Tý) của năm dương lịch."""
    off = _jd_from_date(31, 12, year) - 2415021
    k = int(off / 29.530588853)
    nm = _get_new_moon_day(k, time_zone)
    sun_long = _sun_longitude(nm - 0.5 - time_zone / 24)
    if int(sun_long / (math.pi / 6)) >= 9:
        nm = _get_new_moon_day(k - 1, time_zone)
    return nm


def _get_leap_month_offset(a11: int, time_zone: float) -> int:
    """Xác định tháng nhuận trong năm âm lịch."""
    k = int((a11 - 2415021.076998695) / 29.530588853 + 0.5)
    last = 0
    i = 1
    arc = int(_sun_longitude(_get_new_moon_day(k + i, time_zone) - 0.5 - time_zone / 24) / (math.pi / 6))
    while True:
        last = arc
        i += 1
        arc = int(_sun_longitude(_get_new_moon_day(k + i, time_zone) - 0.5 - time_zone / 24) / (math.pi / 6))
        if arc == last or i >= 14:
            break
    return i - 1


def solar_to_lunar(day: int, month: int, year: int, time_zone: float = 7.0) -> tuple[int, int, int, bool]:
    """
    Chuyển đổi ngày dương lịch sang âm lịch.

    Args:
        day, month, year: Ngày dương lịch
        time_zone: Múi giờ (Việt Nam = 7)

    Returns:
        (ngày_âm, tháng_âm, năm_âm, là_tháng_nhuận)
    """
    day_number = _jd_from_date(day, month, year)
    k = int((day_number - 2415021.076998695) / 29.530588853)
    month_start = _get_new_moon_day(k + 1, time_zone)
    if month_start > day_number:
        month_start = _get_new_moon_day(k, time_zone)

    a11 = _get_lunar_month_11(year, time_zone)
    b11 = a11
    if a11 >= month_start:
        lunar_year = year
        a11 = _get_lunar_month_11(year - 1, time_zone)
    else:
        lunar_year = year + 1
        b11 = _get_lunar_month_11(year + 1, time_zone)

    lunar_day = day_number - month_start + 1
    diff = int((month_start - a11) / 29)
    lunar_leap = False
    lunar_month = diff + 11

    if b11 - a11 > 365:
        leap_month_diff = _get_leap_month_offset(a11, time_zone)
        if diff >= leap_month_diff:
            lunar_month = diff + 10
            if diff == leap_month_diff:
                lunar_leap = True

    if lunar_month > 12:
        lunar_month -= 12
    if lunar_month >= 11 and diff < 4:
        lunar_year -= 1

    return lunar_day, lunar_month, lunar_year, lunar_leap


def lunar_to_solar(
    lunar_day: int,
    lunar_month: int,
    lunar_year: int,
    lunar_leap: bool = False,
    time_zone: float = 7.0,
) -> tuple[int, int, int]:
    """
    Chuyển đổi ngày âm lịch sang dương lịch.

    Args:
        lunar_day, lunar_month, lunar_year: Ngày âm lịch
        lunar_leap: True nếu là tháng nhuận
        time_zone: Múi giờ (Việt Nam = 7)

    Returns:
        (ngày, tháng, năm) dương lịch
    """
    if lunar_month < 11:
        a11 = _get_lunar_month_11(lunar_year - 1, time_zone)
        b11 = _get_lunar_month_11(lunar_year, time_zone)
    else:
        a11 = _get_lunar_month_11(lunar_year, time_zone)
        b11 = _get_lunar_month_11(lunar_year + 1, time_zone)

    k = int(0.5 + (a11 - 2415021.076998695) / 29.530588853)
    off = lunar_month - 11
    if off < 0:
        off += 12

    if b11 - a11 > 365:
        leap_off = _get_leap_month_offset(a11, time_zone)
        leap_month = leap_off - 2
        if leap_month < 0:
            leap_month += 12
        if lunar_leap and lunar_month != leap_month:
            return (0, 0, 0)
        if lunar_leap or off >= leap_off:
            off += 1

    month_start = _get_new_moon_day(k + off, time_zone)
    day, month, year = _date_from_jd(month_start + lunar_day - 1)
    return day, month, year


CAN = ["Giáp", "Ất", "Bính", "Đinh", "Mậu", "Kỷ", "Canh", "Tân", "Nhâm", "Quý"]
CHI = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]
TIET_KHI = [
    "Tiểu Hàn", "Đại Hàn", "Lập Xuân", "Vũ Thủy", "Kinh Trập", "Xuân Phân",
    "Thanh Minh", "Cốc Vũ", "Lập Hạ", "Tiểu Mãn", "Mang Chủng", "Hạ Chí",
    "Tiểu Thử", "Đại Thử", "Lập Thu", "Xử Thử", "Bạch Lộ", "Thu Phân",
    "Hàn Lộ", "Sương Giáng", "Lập Đông", "Tiểu Tuyết", "Đại Tuyết", "Đông Chí",
]


def get_can_chi_year(lunar_year: int) -> str:
    """Trả về tên Can Chi của năm âm lịch. VD: 2024 -> 'Giáp Thìn'."""
    can = CAN[(lunar_year + 6) % 10]
    chi = CHI[(lunar_year + 8) % 12]
    return f"{can} {chi}"


def get_can_chi_day(jd: int) -> str:
    """Trả về tên Can Chi của ngày theo Julian Day."""
    can = CAN[(jd + 9) % 10]
    chi = CHI[(jd + 1) % 12]
    return f"{can} {chi}"


def generate_calendar_csv(start_year: int, end_year: int, filename: str) -> str:
    """
    Sinh bảng chuyển đổi dương-âm lịch cho khoảng năm cho trước và lưu ra CSV.

    Args:
        start_year: Năm dương lịch bắt đầu
        end_year: Năm dương lịch kết thúc (bao gồm)
        filename: Tên file CSV, vd: 'lunar_calendar_2020_2030.csv'

    Returns:
        Đường dẫn tuyệt đối của file đã lưu
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DIR, filename)

    fieldnames = [
        "solar_date", "solar_day", "solar_month", "solar_year",
        "lunar_day", "lunar_month", "lunar_year", "is_leap_month",
        "can_chi_year", "can_chi_day",
    ]

    current = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    count = 0

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        while current <= end:
            d, m, y = current.day, current.month, current.year
            ld, lm, ly, leap = solar_to_lunar(d, m, y)
            jd = _jd_from_date(d, m, y)
            writer.writerow({
                "solar_date": current.isoformat(),
                "solar_day": d,
                "solar_month": m,
                "solar_year": y,
                "lunar_day": ld,
                "lunar_month": lm,
                "lunar_year": ly,
                "is_leap_month": int(leap),
                "can_chi_year": get_can_chi_year(ly),
                "can_chi_day": get_can_chi_day(jd),
            })
            current += timedelta(days=1)
            count += 1

    print(f"Đã lưu {count} ngày ({start_year}-{end_year}) vào {filepath}")
    return filepath


if __name__ == "__main__":
    today = date.today()
    d, m, y = today.day, today.month, today.year
    ld, lm, ly, leap = solar_to_lunar(d, m, y)
    leap_str = " (nhuận)" if leap else ""
    print(f"Hôm nay dương lịch: {d}/{m}/{y}")
    print(f"Âm lịch: {ld}/{lm}{leap_str}/{ly} - {get_can_chi_year(ly)}")
    jd = _jd_from_date(d, m, y)
    print(f"Can Chi ngày: {get_can_chi_day(jd)}")

    print("\nSinh bảng chuyển đổi lịch 2000-2030...")
    generate_calendar_csv(2000, 2030, "lunar_calendar_2000_2030.csv")
