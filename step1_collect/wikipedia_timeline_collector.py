"""
Thu thập facts occurredAt từ trang wikitext dạng niên biểu:
  - "Niên biểu lịch sử Việt Nam" (định dạng '''NĂM:''' mô tả [[liên kết]])
  - Các trang thuộc Thể loại:Sự kiện lịch sử Việt Nam (bài viết đơn lẻ,
    lấy ngày/năm từ đoạn mở đầu bằng regex ngày tháng tiếng Việt)

QUY ƯỚC HEURISTIC (ghi rõ vì không có nhãn địa điểm tường minh trong niên biểu):
  - h (thực thể sự kiện)  = liên kết wiki đầu tiên xuất hiện trong mô tả
  - t (địa điểm)          = liên kết wiki đứng ngay sau các từ khóa định vị
                            ("tại", "kinh đô", "thủ đô", "trên sông", ...);
                            nếu không tìm thấy, fallback = "Việt Nam" (Q881)
  - Năm TCN (trước Công Nguyên) được quy ước là số âm.
  - Dòng lồng (bắt đầu bằng ':') kế thừa năm của dòng '''NĂM''' đứng trước
    (không có mô tả), dùng làm mốc năm chung cho các sự kiện trong năm đó.

Output: data/raw/historical_events_timeline.csv (định dạng 6-ngôi)
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

TIMELINE_PAGE = "Niên biểu lịch sử Việt Nam"
EVENT_CATEGORY = "Sự kiện lịch sử Việt Nam"
VN_DEFAULT_LABEL = "Việt Nam"

LOCATION_MARKERS = [
    "kinh đô tại", "thủ đô tại", "đặt kinh đô", "đặt thủ đô",
    "tại", "trên sông", "trên", "ở",
]

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]")
# Dòng dạng "'''<date span>:''' <mô tả>"
DATED_LINE_RE = re.compile(r"^'''([^']+?):'''\s*(.*)$")
# Dòng chỉ có năm, không mô tả: "'''229'''"
YEAR_ONLY_RE = re.compile(r"^'''([^':]+)'''$")
# Dòng dạng "'''<date span>''' <mô tả>" (không có dấu ':' bên trong đậm)
YEAR_DESC_NO_COLON_RE = re.compile(r"^'''([^':]+)'''\s+(.+)$")
SECTION_RE = re.compile(r"^(={2,4})\s*.*\s*\1$")
REF_TAG_RE = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S)


def _clean_wikilinks(text: str) -> str:
    """Loại bỏ cú pháp [[...]] giữ lại nhãn hiển thị, dùng cho mô tả thô."""
    return WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), text)


def _first_wikilink(text: str) -> str:
    m = WIKILINK_RE.search(text)
    if not m:
        return ""
    return (m.group(2) or m.group(1)).strip()


def _location_wikilink(text: str) -> str:
    """Tìm liên kết wiki đứng sau một từ khóa định vị (ranh giới từ), khác với
    liên kết đầu tiên (h); rỗng nếu không thấy."""
    first = WIKILINK_RE.search(text)
    search_start = first.end() if first else 0
    first_label = (first.group(2) or first.group(1)).strip() if first else None

    lowered = text.lower()
    for marker in LOCATION_MARKERS:
        for m_marker in re.finditer(r"(?<!\w)" + re.escape(marker) + r"(?!\w)", lowered):
            idx = m_marker.end()
            if idx < search_start:
                continue
            after = text[idx:idx + 80]
            m = WIKILINK_RE.search(after)
            if m:
                label = (m.group(2) or m.group(1)).strip()
                if label != first_label:
                    return label
    return ""


def parse_year_token(tok: str) -> int | None:
    """'25.000 TCN' -> -25000 ; '905' -> 905 ; 'Thế kỉ VII TCN' -> None (không phải số).
    'tháng'/'Tháng' -> None (biểu thức ngày/tháng, không phải năm — nếu ghép số ngày+tháng
    thành chuỗi chữ số sẽ tạo ra năm giả, vd '22 tháng 12' -> '2212')."""
    tok = tok.strip()
    if "tháng" in tok.lower():
        return None
    is_bce = "TCN" in tok.upper()
    digits = re.sub(r"[^\d]", "", tok)
    if not digits:
        return None
    year = int(digits)
    return -year if is_bce else year


def parse_year_span(span_text: str) -> tuple[int | None, int | None]:
    """'42 - 43' -> (42,43) ; '207 TCN' -> (-207,-207) ; '25.000 TCN–7.000 TCN' -> (-25000,-7000)."""
    parts = re.split(r"\s*[-–]\s*", span_text.strip())
    if len(parts) == 1:
        y = parse_year_token(parts[0])
        return y, y
    y1 = parse_year_token(parts[0])
    y2 = parse_year_token(parts[1])
    if y1 is None or y2 is None:
        return None, None
    return min(y1, y2), max(y1, y2)


class WikipediaTimelineCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ViSTKG-QA-research/1.0 (minhhaivatc@gmail.com)"})

    def _get(self, params: dict) -> dict:
        params["format"] = "json"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(WIKIPEDIA_API_URL, params=params, timeout=TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning(f"Lần thử {attempt}/{MAX_RETRIES} thất bại: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(REQUEST_DELAY * attempt)
        return {}

    def get_wikitext(self, title: str) -> str:
        data = self._get({"action": "parse", "page": title, "prop": "wikitext"})
        return data.get("parse", {}).get("wikitext", {}).get("*", "")

    def get_category_members(self, category: str, limit: int = 500) -> list[str]:
        data = self._get({
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Thể loại:{category}", "cmlimit": limit, "cmnamespace": 0,
        })
        return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]

    def get_intro_text(self, title: str) -> str:
        data = self._get({"action": "query", "prop": "extracts", "explaintext": True,
                           "exintro": True, "titles": title})
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")
        return ""

    # ------------------------------------------------------------------
    # Nguồn 1: trang niên biểu (regex năm/ngày: mô tả)
    # ------------------------------------------------------------------

    def parse_timeline_page(self, wikitext: str) -> list[dict]:
        wikitext = REF_TAG_RE.sub("", wikitext)
        rows = []
        current_year: int | None = None
        skipped_unparsable = 0

        def resolve_span(span_text: str, is_nested: bool) -> tuple[int | None, int | None]:
            """Ưu tiên context năm cha cho biểu thức ngày/tháng (không có năm riêng);
            chỉ dùng năm parse trực tiếp từ span_text khi nó thực sự chứa một năm."""
            has_own_year = "tháng" not in span_text.lower()
            if is_nested and current_year is not None and not has_own_year:
                return current_year, current_year
            y_start, y_end = parse_year_span(span_text)
            if y_start is None and is_nested and current_year is not None:
                return current_year, current_year
            return y_start, y_end

        def emit(span_text: str, desc: str, is_nested: bool) -> bool:
            """True nếu đã sinh được 1 fact."""
            nonlocal skipped_unparsable
            y_start, y_end = resolve_span(span_text, is_nested)
            if y_start is None:
                skipped_unparsable += 1
                return False
            desc_clean = _clean_wikilinks(desc)
            if not desc_clean.strip():
                return False
            h_label = _first_wikilink(desc)
            if not h_label:
                return False
            t_label = _location_wikilink(desc) or VN_DEFAULT_LABEL
            rows.append({
                "h": h_label, "h_label": h_label,
                "r": "occurredAt",
                "t": t_label, "t_label": t_label,
                "tau_start": y_start, "tau_end": y_end,
                "l_h_lat": "", "l_h_lon": "", "l_t_lat": "", "l_t_lon": "",
                "vi_context": desc_clean.strip(),
                "source": "Wikipedia-Timeline",
            })
            return True

        for raw_line in wikitext.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if SECTION_RE.match(line):
                current_year = None
                continue

            stripped_colon = line.lstrip(":").strip()
            is_nested = line.startswith(":")

            m = DATED_LINE_RE.match(stripped_colon)
            if m:
                span_text, desc = m.group(1), m.group(2)
                if emit(span_text, desc, is_nested) and not is_nested:
                    y_start, _ = resolve_span(span_text, is_nested)
                    current_year = y_start
                continue

            m2 = YEAR_ONLY_RE.match(stripped_colon)
            if m2:
                y_start, _ = parse_year_span(m2.group(1))
                if y_start is not None:
                    current_year = y_start
                continue

            m3 = YEAR_DESC_NO_COLON_RE.match(stripped_colon)
            if m3:
                span_text, desc = m3.group(1), m3.group(2)
                y_start, _ = parse_year_span(span_text)
                if y_start is None:
                    skipped_unparsable += 1
                    continue
                if not is_nested:
                    current_year = y_start
                emit(span_text, desc, is_nested)
                continue

        logger.info(f"Timeline: {len(rows)} facts trích được, {skipped_unparsable} dòng bỏ qua (không parse được năm)")
        return rows

    # ------------------------------------------------------------------
    # Nguồn 2: Thể loại:Sự kiện lịch sử Việt Nam (bài viết đơn lẻ)
    # ------------------------------------------------------------------

    def parse_category_events(self, titles: list[str]) -> list[dict]:
        date_patterns = [
            re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{3,4})", re.I),
            re.compile(r"(\d{1,2})/(\d{1,2})/(\d{3,4})"),
            re.compile(r"năm\s+(\d{3,4})", re.I),
            re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b"),
        ]
        rows = []
        for title in titles:
            intro = self.get_intro_text(title)
            time.sleep(REQUEST_DELAY)
            if not intro:
                continue
            year = None
            for pat in date_patterns:
                m = pat.search(intro)
                if m:
                    groups = [g for g in m.groups() if g]
                    year = int(groups[-1])
                    break
            if year is None:
                continue
            t_label = _location_wikilink(intro) or VN_DEFAULT_LABEL
            rows.append({
                "h": title, "h_label": title,
                "r": "occurredAt",
                "t": t_label, "t_label": t_label,
                "tau_start": year, "tau_end": year,
                "l_h_lat": "", "l_h_lon": "", "l_t_lat": "", "l_t_lon": "",
                "vi_context": intro[:300],
                "source": "Wikipedia-EventCategory",
            })
        logger.info(f"Category events: {len(rows)}/{len(titles)} bài trích được năm")
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
    collector = WikipediaTimelineCollector()

    print("-" * 50)
    print(f"1/2  Parse trang '{TIMELINE_PAGE}'...")
    wikitext = collector.get_wikitext(TIMELINE_PAGE)
    timeline_rows = collector.parse_timeline_page(wikitext)
    print(f"     -> {len(timeline_rows)} facts")

    print("-" * 50)
    print(f"2/2  Thu thập Thể loại:{EVENT_CATEGORY}...")
    titles = collector.get_category_members(EVENT_CATEGORY)
    print(f"     -> {len(titles)} bài viết trong thể loại")
    category_rows = collector.parse_category_events(titles)
    print(f"     -> {len(category_rows)} facts")

    all_rows = timeline_rows + category_rows
    path = collector.save_csv(all_rows, "historical_events_timeline.csv")
    print(f"Tong hop {len(all_rows)} facts -> {path}")

    n_fallback = sum(1 for r in all_rows if r["t_label"] == VN_DEFAULT_LABEL)
    print(f"Facts fallback ve dia diem '{VN_DEFAULT_LABEL}' (khong xac dinh duoc dia diem cu the): "
          f"{n_fallback}/{len(all_rows)} ({n_fallback/len(all_rows)*100:.1f}%)")
