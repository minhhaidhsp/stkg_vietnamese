"""
Thu thập ảnh cho các thực thể trong knowledge graph.
Nguồn ưu tiên: Wikidata P18 → Wikipedia tiếng Việt → bỏ qua.
"""

import json
import logging
import os
import random
import sys
import time
import urllib.parse

import requests
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGES_DIR, VISUAL_DIR, IMAGE_WIDTH, REQUEST_DELAY, TIMEOUT, MAX_RETRIES

URL_CACHE_FILE = os.path.join(VISUAL_DIR, "url_cache.json")

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

    def download_image(self, url: str, dest_path: str) -> bool:
        """Tải ảnh về, resize về IMAGE_WIDTH, lưu dạng JPEG.
        Dùng requests.get() tươi (không session) + random delay + retry thông minh."""
        WAIT_TIMES = [30, 60, 120]   # giây chờ khi gặp 429
        headers = {"User-Agent": "stkg_vietnamese/1.0 (research)"}

        for attempt in range(MAX_RETRIES):
            # Delay ngẫu nhiên trước mỗi request (2-5 giây)
            time.sleep(random.uniform(2, 5))
            try:
                resp = requests.get(url, timeout=TIMEOUT, headers=headers)

                if resp.status_code == 200:
                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                    ratio = IMAGE_WIDTH / max(img.width, 1)
                    new_h = max(1, int(img.height * ratio))
                    img   = img.resize((IMAGE_WIDTH, new_h), Image.LANCZOS)
                    img.save(dest_path, "JPEG", quality=85)
                    return True

                elif resp.status_code == 429:
                    wait = WAIT_TIMES[attempt]
                    logger.info(f"  [429] Rate-limit, cho {wait}s... (attempt {attempt+1})")
                    time.sleep(wait)

                else:
                    logger.warning(f"  [HTTP {resp.status_code}] {url}")
                    break

            except Exception as e:
                logger.warning(f"  Download error attempt {attempt+1}: {e}")
                break

        return False

    # ------------------------------------------------------------------
    # URL cache — tránh fetch lại Wikidata/Wikipedia mỗi lần chạy
    # ------------------------------------------------------------------

    def _load_url_cache(self) -> dict[str, str]:
        if os.path.exists(URL_CACHE_FILE):
            with open(URL_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_url_cache(self, cache: dict[str, str]) -> None:
        with open(URL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Pipeline chính
    # ------------------------------------------------------------------

    def collect_batch(
        self,
        entities: list[dict],    # list of {"qid": ..., "label": ...}
        skip_existing: bool = True,
        skip_download: bool = False,
        deadline: float | None = None,   # time.time() tuyệt đối — dừng tải ảnh khi vượt qua (L1, đóng khung 24h)
    ) -> dict[str, str]:
        """
        Tải ảnh cho nhiều thực thể.
        URL cache giúp bỏ qua giai đoạn fetch API trong các lần chạy sau.

        Args:
            skip_download: Nếu True, bỏ qua download thực tế.
                           Trả về {qid: ""} cho các QID có URL trong cache
                           (has_image=True dù chưa có file thực).
        Returns:
            dict {qid: local_image_path}. Path rỗng "" nếu skip_download=True.
        """
        qids = [e["qid"] for e in entities]
        label_map = {e["qid"]: e["label"] for e in entities}

        # Tìm QID đã có ảnh local
        if skip_existing:
            already = {q: self._local_path(q) for q in qids if os.path.exists(self._local_path(q))}
            qids_to_fetch = [q for q in qids if q not in already]
            logger.info(f"Skip {len(already)} cached images, need to process {len(qids_to_fetch)}")
        else:
            already = {}
            qids_to_fetch = qids

        # Load URL cache
        url_cache = self._load_url_cache()
        cached_urls = {q: url_cache[q] for q in qids_to_fetch if q in url_cache}
        qids_need_fetch = [q for q in qids_to_fetch if q not in url_cache]
        logger.info(f"URL cache: {len(cached_urls)} hits, {len(qids_need_fetch)} need API fetch")

        # Fetch URL từ Wikidata P18 chỉ cho QID chưa có trong cache
        if qids_need_fetch:
            new_wikidata = self.get_wikidata_image_urls(qids_need_fetch)
            logger.info(f"Wikidata P18: {len(new_wikidata)}/{len(qids_need_fetch)} QIDs co anh")

            # Fallback Wikipedia cho QID chưa có ảnh
            missing = [q for q in qids_need_fetch if q not in new_wikidata]
            new_wiki: dict[str, str] = {}
            for qid in missing:
                label = label_map.get(qid, "")
                wiki_url = self.get_wikipedia_image_url(label)
                if wiki_url:
                    new_wiki[qid] = wiki_url
                    time.sleep(REQUEST_DELAY)

            # Merge vào cache và lưu (cả QID không có ảnh cũng lưu "" để không fetch lại)
            for q in qids_need_fetch:
                if q in new_wikidata:
                    url_cache[q] = new_wikidata[q]
                elif q in new_wiki:
                    url_cache[q] = new_wiki[q]
                else:
                    url_cache[q] = ""   # đánh dấu "đã fetch, không có ảnh"
            self._save_url_cache(url_cache)
            logger.info(f"URL cache da luu: {len(url_cache)} entries -> {URL_CACHE_FILE}")
        else:
            logger.info("Bo qua fetch API — dung 100% URL cache")

        # Tổng hợp url_map (bỏ qua QID không có URL)
        qids_set = set(qids_to_fetch)
        url_map = {q: url for q, url in url_cache.items() if q in qids_set and url}
        logger.info(f"Total URLs can download: {len(url_map)}")

        # Mode skip_download: chỉ dùng URL cache, không tải file
        if skip_download:
            result = dict(already)
            for qid in url_map:
                result[qid] = ""   # "" = có URL nhưng chưa tải
            logger.info(f"skip_download=True: {len(url_map)} entities co URL (chua tai anh)")
            return result

        # Tải ảnh — random delay, circuit breaker nếu quá nhiều 429
        result = dict(already)
        ok = fail = consec_fail = 0
        items = list(url_map.items())
        stopped_by_deadline = False
        for i, (qid, url) in enumerate(items, 1):
            if deadline is not None and time.time() >= deadline:
                logger.warning(f"  Het han 24h (deadline) o {i}/{len(items)} — dung lai, dong bang ket qua da co.")
                stopped_by_deadline = True
                break
            dest = self._local_path(qid)
            if os.path.exists(dest):           # skip nếu đã có từ lần trước
                result[qid] = dest
                ok += 1
                consec_fail = 0
            elif self.download_image(url, dest):
                result[qid] = dest
                ok += 1
                consec_fail = 0
            else:
                fail += 1
                consec_fail += 1
                # Circuit breaker: 5 lần fail liên tiếp → nghỉ 5 phút
                if consec_fail >= 5:
                    logger.warning(f"  {consec_fail} lan fail lien tiep, nghi 5 phut...")
                    time.sleep(300)
                    consec_fail = 0
            if i % 20 == 0 or i == len(items):
                logger.info(f"  Download {i}/{len(items)}: {ok} ok, {fail} failed")
        logger.info(f"Download xong: {ok} ok, {fail} failed. Total with image: {len(result)}. "
                    f"Dung do het han 24h: {stopped_by_deadline}")
        self._last_run_stopped_by_deadline = stopped_by_deadline
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
