"""
Thu thập facts locatedIn từ trang "Danh sách Di tích quốc gia Việt Nam".

Cấu trúc trang (xác nhận qua wikitext thật):
  ==Vùng==
  ===Tỉnh/Thành===
  {| ... bảng wiki ...
  |- (dòng phân cách)
  ! cột tiêu đề (Di tích | Địa bàn | Phân loại | Năm Công nhận | Ghi chú)
  |-
  | [[Tên di tích]]|| [[Địa bàn]]|| Phân loại|| Năm công nhận|| Ghi chú<ref>...</ref>
  ...

Mỗi dòng dữ liệu là MỘT dòng wikitext bắt đầu bằng "|" (không phải "|-"),
các ô cách nhau bằng "||".

Ánh xạ 6-ngôi:
  h = tên di tích (liên kết wiki cột 1)
  r = "locatedIn"
  t = tên tỉnh/thành (theo === header) -- dùng cấp tỉnh thay vì địa bàn xã/phường
      cột 2 để đảm bảo geocode ổn định ở Bước 2 (danh sách tỉnh hữu hạn, đã biết)
  tau_start = năm công nhận (năm 4 chữ số đầu tiên tìm thấy trong cột 4)
  vi_context = địa bàn cấp xã/phường (cột 2) + phân loại (cột 3), để tham khảo

Output: data/raw/heritage_sites.csv (định dạng 6-ngôi)
"""

import csv
import logging
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIKIPEDIA_API_URL, REQUEST_DELAY, MAX_RETRIES, TIMEOUT, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HERITAGE_PAGE = "Danh sách Di tích quốc gia Việt Nam"

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")  # cho phép '#' (anchor mục) trong tên trang
EXTERNAL_LINK_RE = re.compile(r"\[https?://\S+\s+([^\]]+)\]|\[https?://\S+\]")  # [URL Nhãn] hoặc [URL]
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")  # {{Webarchive|...}}, {{Liên kết hỏng|...}} — không lồng nhau
REF_TAG_RE = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S)
REGION_RE = re.compile(r"^==([^=].*?)==$")
PROVINCE_RE = re.compile(r"^===([^=].*?)===$")
YEAR_RE = re.compile(r"(\d{4})")


def _cell_label(cell: str) -> str:
    """Lấy nhãn hiển thị của ô: ưu tiên liên kết wiki, else text thô.
    Nếu không có '|' hiển thị riêng, bỏ phần '#anchor' (mục neo) khỏi tên
    trang — đúng cách MediaWiki hiển thị [[Page#Section]] thành "Page".
    Cuối cùng dọn phòng thủ mọi '[[' / ']]' còn sót (vd lỗi cú pháp wikitext
    gốc thiếu '[[' mở — xác nhận thật trên trang nguồn, không phải lỗi parser
    của ta, xem docs/DECISIONS.md)."""
    cell = cell.strip().lstrip("|").strip()
    cell = TEMPLATE_RE.sub("", cell)              # bỏ {{Webarchive|...}}, {{Liên kết hỏng|...}}
    cell = EXTERNAL_LINK_RE.sub(r"\1", cell)       # [URL Nhãn] -> Nhãn ; [URL] -> ""
    m = WIKILINK_RE.search(cell)
    if m:
        label = m.group(2) or m.group(1).split("#")[0]
        return label.strip()
    cleaned = re.sub(r"<[^>]+>", "", cell).strip()
    return cleaned.replace("[[", "").replace("]]", "").strip()


def _extract_year(cell: str) -> int | None:
    m = YEAR_RE.search(cell)
    return int(m.group(1)) if m else None


class WikipediaHeritageCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ViSTKG-QA-research/1.0 (minhhaivatc@gmail.com)"})

    def get_wikitext(self, title: str) -> str:
        params = {"action": "parse", "page": title, "prop": "wikitext", "format": "json"}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(WIKIPEDIA_API_URL, params=params, timeout=TIMEOUT)
                resp.raise_for_status()
                return resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
            except requests.RequestException as e:
                logger.warning(f"Lần thử {attempt}/{MAX_RETRIES} thất bại: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(REQUEST_DELAY * attempt)
        return ""

    def parse_heritage_page(self, wikitext: str) -> list[dict]:
        wikitext = REF_TAG_RE.sub("", wikitext)
        rows = []
        current_region = ""
        current_province = ""
        skipped_bad_cells = 0
        skipped_no_year = 0

        for raw_line in wikitext.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                continue

            m_region = REGION_RE.match(line.strip())
            if m_region:
                current_region = m_region.group(1).strip()
                current_province = ""
                continue

            m_prov = PROVINCE_RE.match(line.strip())
            if m_prov:
                current_province = m_prov.group(1).strip()
                continue

            if not line.startswith("|") or line.startswith("|-") or line.startswith("|}") or line.startswith("!"):
                continue
            if "||" not in line:
                continue
            if not current_province:
                continue

            cells = line.split("||")
            if len(cells) < 4:
                skipped_bad_cells += 1
                continue

            site_name = _cell_label(cells[0])
            commune = _cell_label(cells[1])
            category = _cell_label(cells[2])
            year = _extract_year(cells[3])

            if not site_name:
                skipped_bad_cells += 1
                continue
            if year is None:
                skipped_no_year += 1

            rows.append({
                "h": site_name, "h_label": site_name,
                "r": "locatedIn",
                "t": current_province, "t_label": current_province,
                "tau_start": year if year is not None else "",
                "tau_end": year if year is not None else "",
                "l_h_lat": "", "l_h_lon": "", "l_t_lat": "", "l_t_lon": "",
                "vi_context": f"{commune}; {category}; vùng {current_region}".strip("; "),
                "source": "Wikipedia-Heritage",
            })

        logger.info(f"Heritage: {len(rows)} facts, "
                    f"{skipped_bad_cells} dòng bỏ qua (thiếu cột/tên), "
                    f"{skipped_no_year} facts không xác định được năm công nhận")
        return rows

    def save_csv(self, rows: list[dict], filename: str) -> str:
        if not rows:
            logger.warning(f"Không có dữ liệu để lưu vào {filename}.")
            return ""
        os.makedirs(RAW_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DIR, filename)
        cols = ["h", "h_label", "r", "t", "t_label", "tau_start", "tau_end",
                "l_h_lat", "l_h_lon", "l_t_lat", "l_t_lon", "vi_context", "source"]
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Đã lưu {len(rows)} dòng vào {filepath}")
        return filepath


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    collector = WikipediaHeritageCollector()

    print("-" * 50)
    print(f"Parse trang '{HERITAGE_PAGE}'...")
    wikitext = collector.get_wikitext(HERITAGE_PAGE)
    rows = collector.parse_heritage_page(wikitext)
    print(f"-> {len(rows)} facts")

    path = collector.save_csv(rows, "heritage_sites.csv")
    print(f"Saved -> {path}")

    provinces = {r["t_label"] for r in rows}
    print(f"So tinh/thanh xuat hien: {len(provinces)}")
    n_no_year = sum(1 for r in rows if r["tau_start"] == "")
    print(f"Facts khong co nam cong nhan: {n_no_year}/{len(rows)} ({n_no_year/len(rows)*100:.1f}%)")
