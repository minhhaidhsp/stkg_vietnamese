import csv
import time
import logging
import os
import sys
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIKIPEDIA_API_URL, WIKIPEDIA_LANG, REQUEST_DELAY, MAX_RETRIES, TIMEOUT, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class WikipediaCollector:
    """Thu thập dữ liệu từ Wikipedia tiếng Việt."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "stkg_vietnamese/1.0 (research project)"})

    def _get(self, params: dict) -> dict:
        """Gọi Wikipedia API với retry."""
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

    def get_page_content(self, title: str) -> str:
        """Lấy nội dung văn bản của một trang Wikipedia."""
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "titles": title,
        }
        data = self._get(params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                logger.warning(f"Trang '{title}' không tồn tại.")
                return ""
            return page.get("extract", "")
        return ""

    def get_page_categories(self, title: str) -> list[str]:
        """Lấy danh sách thể loại của một trang Wikipedia."""
        params = {
            "action": "query",
            "prop": "categories",
            "titles": title,
            "cllimit": "max",
        }
        data = self._get(params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            return [cat["title"] for cat in page.get("categories", [])]
        return []

    def search_pages(self, query: str, limit: int = 10) -> list[dict]:
        """Tìm kiếm trang Wikipedia theo từ khóa."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srprop": "snippet|titlesnippet",
        }
        data = self._get(params)
        return data.get("query", {}).get("search", [])

    def get_pages_in_category(self, category: str, limit: int = 100) -> list[str]:
        """Lấy danh sách trang trong một thể loại Wikipedia."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Thể loại:{category}",
            "cmlimit": limit,
            "cmtype": "page",
        }
        data = self._get(params)
        members = data.get("query", {}).get("categorymembers", [])
        return [m["title"] for m in members]

    def save_search_to_csv(self, results: list[dict], filename: str) -> str:
        """
        Lưu kết quả tìm kiếm Wikipedia ra CSV.

        Returns:
            Đường dẫn tuyệt đối của file đã lưu
        """
        if not results:
            logger.warning("Không có kết quả để lưu.")
            return ""

        os.makedirs(RAW_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DIR, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "snippet"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
                })

        logger.info(f"Đã lưu {len(results)} dòng vào {filepath}")
        return filepath

    def save_pages_to_csv(self, pages_data: list[dict], filename: str) -> str:
        """
        Lưu danh sách trang kèm nội dung ra CSV.

        Args:
            pages_data: list of {"title": ..., "content": ..., "categories": [...]}
        """
        if not pages_data:
            logger.warning("Không có trang để lưu.")
            return ""

        os.makedirs(RAW_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DIR, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "categories", "content"])
            writer.writeheader()
            for p in pages_data:
                writer.writerow({
                    "title": p.get("title", ""),
                    "categories": "|".join(p.get("categories", [])),
                    "content": p.get("content", "")[:2000],  # giới hạn 2000 ký tự
                })

        logger.info(f"Đã lưu {len(pages_data)} trang vào {filepath}")
        return filepath


if __name__ == "__main__":
    collector = WikipediaCollector()

    # --- Tìm kiếm và lưu kết quả ---
    keywords = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Huế", "Cần Thơ"]
    all_search = []
    for kw in keywords:
        print(f"Tìm kiếm '{kw}'...")
        results = collector.search_pages(kw, limit=20)
        all_search.extend(results)
        time.sleep(REQUEST_DELAY)

    path = collector.save_search_to_csv(all_search, "wikipedia_search_results.csv")
    print(f"Lưu {len(all_search)} kết quả tìm kiếm -> {path}")

    # --- Lấy nội dung các trang chính và lưu ---
    titles = collector.get_pages_in_category("Tỉnh Việt Nam", limit=63)
    print(f"\nThu thập nội dung {len(titles)} trang tỉnh thành...")
    pages_data = []
    for title in titles:
        content = collector.get_page_content(title)
        categories = collector.get_page_categories(title)
        pages_data.append({"title": title, "content": content, "categories": categories})
        time.sleep(REQUEST_DELAY)

    path = collector.save_pages_to_csv(pages_data, "wikipedia_viet_provinces.csv")
    print(f"Lưu {len(pages_data)} trang tỉnh thành -> {path}")
