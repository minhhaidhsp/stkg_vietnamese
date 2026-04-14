"""
Thu thập ảnh cho các thực thể trong knowledge graph.
Nguồn ưu tiên: Wikidata P18 → Wikipedia tiếng Việt → bỏ qua.
"""

import logging
import os
import sys
import time
import urllib.parse

import requests
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGES_DIR, IMAGE_WIDTH, REQUEST_DELAY, TIMEOUT, MAX_RETRIES

logger = logging.getLogger(__name__)

WIKIDATA_API  = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://vi.wikipedia.org/w/api.php"
COMMONS_BASE  = "https://commons.wikimedia.org/wiki/Special:FilePath"
BATCH_SIZE    = 50   # Wikidata cho phép tối đa 50 QID/request


class ImageCollector:
    """Tải ảnh đại diện từ Wikidata (P18) hoặc Wikipedia về thư mục local."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "stkg_vietnamese/1.0 (research; contact: stkg@example.com)"
        })
        os.makedirs(IMAGES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Lấy URL ảnh từ Wikidata P18
    # ------------------------------------------------------------------

    def get_wikidata_image_urls(self, qids: list[str]) -> dict[str, str]:
        """
        Lấy URL ảnh P18 cho danh sách QIDs (batch).

        Returns:
            dict {qid: wikimedia_url}
        """
        result: dict[str, str] = {}
        for i in range(0, len(qids), BATCH_SIZE):
            batch = qids[i: i + BATCH_SIZE]
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "claims",
                "format": "json",
            }
            try:
                resp = self.session.get(WIKIDATA_API, params=params, timeout=TIMEOUT)
                resp.raise_for_status()
                entities = resp.json().get("entities", {})
                for qid, entity in entities.items():
                    p18 = entity.get("claims", {}).get("P18", [])
                    if p18:
                        filename = p18[0]["mainsnak"]["datavalue"]["value"]
                        encoded  = urllib.parse.quote(filename.replace(" ", "_"))
                        result[qid] = f"{COMMONS_BASE}/{encoded}?width={IMAGE_WIDTH}"
            except Exception as e:
                logger.warning(f"Wikidata image batch error: {e}")
            time.sleep(REQUEST_DELAY)
        return result

    def get_wikipedia_image_url(self, title: str) -> str | None:
        """Lấy ảnh thumbnail từ trang Wikipedia tiếng Việt."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "piprop": "original",
            "format": "json",
        }
        try:
            resp = self.session.get(WIKIPEDIA_API, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                original = page.get("original", {})
                if original.get("source"):
                    return original["source"]
        except Exception as e:
            logger.debug(f"Wikipedia image error cho '{title}': {e}")
        return None

    # ------------------------------------------------------------------
    # Tải & lưu ảnh
    # ------------------------------------------------------------------

    def _local_path(self, qid: str) -> str:
        return os.path.join(IMAGES_DIR, f"{qid}.jpg")

    def download(self, url: str, dest_path: str) -> bool:
        """Tải ảnh về, resize về IMAGE_WIDTH, lưu dạng JPEG.
        Dùng requests.get() tươi (không session) để tránh bị rate-limit."""
        headers = {"User-Agent": "stkg_vietnamese/1.0 (research)"}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(url, timeout=TIMEOUT, headers=headers)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                # Resize giữ tỷ lệ
                ratio  = IMAGE_WIDTH / max(img.width, 1)
                new_h  = max(1, int(img.height * ratio))
                img    = img.resize((IMAGE_WIDTH, new_h), Image.LANCZOS)
                img.save(dest_path, "JPEG", quality=85)
                return True
            except Exception as e:
                logger.warning(f"Download attempt {attempt} failed for {url}: {e}")
                if attempt < MAX_RETRIES:
                    # 429 = rate limit → backoff dài: 60s, 120s
                    if "429" in str(e):
                        wait = 60 * attempt
                        logger.info(f"  Rate-limit 429, chờ {wait}s...")
                    else:
                        wait = REQUEST_DELAY * attempt
                    time.sleep(wait)
        return False

    # ------------------------------------------------------------------
    # Pipeline chính
    # ------------------------------------------------------------------

    def collect_batch(
        self,
        entities: list[dict],    # list of {"qid": ..., "label": ...}
        skip_existing: bool = True,
    ) -> dict[str, str]:
        """
        Tải ảnh cho nhiều thực thể.

        Returns:
            dict {qid: local_image_path} chỉ gồm các QID có ảnh thành công.
        """
        qids = [e["qid"] for e in entities]
        label_map = {e["qid"]: e["label"] for e in entities}

        # Tìm QID đã có ảnh local
        if skip_existing:
            already = {q: self._local_path(q) for q in qids if os.path.exists(self._local_path(q))}
            qids_to_fetch = [q for q in qids if q not in already]
            logger.info(f"Skip {len(already)} cached, fetching {len(qids_to_fetch)} new")
        else:
            already = {}
            qids_to_fetch = qids

        # Lấy URL từ Wikidata P18
        url_map = self.get_wikidata_image_urls(qids_to_fetch)
        logger.info(f"Wikidata P18: {len(url_map)}/{len(qids_to_fetch)} QIDs có ảnh")

        # Fallback Wikipedia cho QID chưa có ảnh
        missing = [q for q in qids_to_fetch if q not in url_map]
        for qid in missing:
            label = label_map.get(qid, "")
            wiki_url = self.get_wikipedia_image_url(label)
            if wiki_url:
                url_map[qid] = wiki_url
                time.sleep(REQUEST_DELAY)

        logger.info(f"Total URLs sau fallback: {len(url_map)}")

        # Tải ảnh — delay 2s giữa các ảnh, circuit breaker nếu quá nhiều 429
        result = dict(already)
        ok = fail = consec_fail = 0
        items = list(url_map.items())
        for i, (qid, url) in enumerate(items, 1):
            dest = self._local_path(qid)
            if os.path.exists(dest):           # skip nếu đã có từ lần trước
                result[qid] = dest
                ok += 1
                consec_fail = 0
            elif self.download(url, dest):
                result[qid] = dest
                ok += 1
                consec_fail = 0
            else:
                fail += 1
                consec_fail += 1
                # Circuit breaker: 5 lần fail liên tiếp → nghỉ 5 phút
                if consec_fail >= 5:
                    logger.warning(f"  {consec_fail} lần fail liên tiếp, nghỉ 5 phút...")
                    time.sleep(300)
                    consec_fail = 0
            time.sleep(2)                      # rate-limit: tối đa 0.5 req/s
            if i % 20 == 0 or i == len(items):
                logger.info(f"  Download {i}/{len(items)}: {ok} ok, {fail} failed")
        logger.info(f"Download xong: {ok} ok, {fail} failed. Total with image: {len(result)}")
        return result


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    collector = ImageCollector()
    test = [
        {"qid": "Q36014",  "label": "Hồ Chí Minh"},
        {"qid": "Q1858",   "label": "Hà Nội"},
        {"qid": "Q178561", "label": "Điện Biên Phủ"},
    ]
    images = collector.collect_batch(test)
    for qid, path in images.items():
        print(f"{qid} -> {path}")
