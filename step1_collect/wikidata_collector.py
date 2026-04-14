import csv
import time
import logging
import os
import sys
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIKIDATA_SPARQL_ENDPOINT, REQUEST_DELAY, MAX_RETRIES, TIMEOUT, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SPARQL queries — định dạng 6-ngôi (h, h_label, r, t, t_label, τ, l_h, l_t)
# ---------------------------------------------------------------------------

SPARQL_HISTORICAL_FIGURES = """
SELECT DISTINCT
  (STRAFTER(STR(?h), "entity/") AS ?h_id) ?h_label
  ?r
  (STRAFTER(STR(?t), "entity/") AS ?t_id) ?t_label
  ?tau_start
  ?l_h_lat ?l_h_lon
  ?l_t_lat ?l_t_lon
WHERE {
  {
    # Nơi sinh
    ?h wdt:P31 wd:Q5 ; wdt:P27 wd:Q881 .
    ?h wdt:P19 ?t .
    ?h wdt:P569 ?birth_date .
    BIND(YEAR(?birth_date) AS ?tau_start)
    BIND("bornIn" AS ?r)
    ?h rdfs:label ?h_label . FILTER(LANG(?h_label) = "vi")
    ?t rdfs:label ?t_label . FILTER(LANG(?t_label) = "vi")
    OPTIONAL {
      ?t p:P625/psv:P625 ?tn .
      ?tn wikibase:geoLatitude ?l_t_lat ; wikibase:geoLongitude ?l_t_lon .
    }
    OPTIONAL {
      ?h p:P625/psv:P625 ?hn .
      ?hn wikibase:geoLatitude ?l_h_lat ; wikibase:geoLongitude ?l_h_lon .
    }
  }
  UNION
  {
    # Nơi mất
    ?h wdt:P31 wd:Q5 ; wdt:P27 wd:Q881 .
    ?h wdt:P20 ?t .
    ?h wdt:P570 ?death_date .
    BIND(YEAR(?death_date) AS ?tau_start)
    BIND("diedIn" AS ?r)
    ?h rdfs:label ?h_label . FILTER(LANG(?h_label) = "vi")
    ?t rdfs:label ?t_label . FILTER(LANG(?t_label) = "vi")
    OPTIONAL {
      ?t p:P625/psv:P625 ?tn .
      ?tn wikibase:geoLatitude ?l_t_lat ; wikibase:geoLongitude ?l_t_lon .
    }
    OPTIONAL {
      ?h p:P625/psv:P625 ?hn .
      ?hn wikibase:geoLatitude ?l_h_lat ; wikibase:geoLongitude ?l_h_lon .
    }
  }
}
LIMIT 500
"""

SPARQL_LANDMARKS = """
SELECT DISTINCT
  (STRAFTER(STR(?h), "entity/") AS ?h_id) ?h_label
  ?r
  (STRAFTER(STR(?t), "entity/") AS ?t_id) ?t_label
  ?tau_start
  ?l_h_lat ?l_h_lon
  ?l_t_lat ?l_t_lon
WHERE {
  {
    { ?h wdt:P31 wd:Q41176 . }   UNION
    { ?h wdt:P31 wd:Q839954 . }  UNION
    { ?h wdt:P31 wd:Q33506 . }   UNION
    { ?h wdt:P31 wd:Q44613 . }   UNION
    { ?h wdt:P31 wd:Q16748862 . }
  }
  BIND("locatedIn" AS ?r)
  ?h wdt:P17 wd:Q881 .
  ?h wdt:P131 ?t .
  ?h rdfs:label ?h_label . FILTER(LANG(?h_label) = "vi")
  ?t rdfs:label ?t_label . FILTER(LANG(?t_label) = "vi")
  OPTIONAL { ?h wdt:P571 ?inception . BIND(YEAR(?inception) AS ?tau_start) }
  OPTIONAL {
    ?h p:P625/psv:P625 ?hn .
    ?hn wikibase:geoLatitude ?l_h_lat ; wikibase:geoLongitude ?l_h_lon .
  }
  OPTIONAL {
    ?t p:P625/psv:P625 ?tn .
    ?tn wikibase:geoLatitude ?l_t_lat ; wikibase:geoLongitude ?l_t_lon .
  }
}
LIMIT 500
"""

SPARQL_HISTORICAL_EVENTS = """
SELECT DISTINCT
  (STRAFTER(STR(?h), "entity/") AS ?h_id) ?h_label
  ?r
  (STRAFTER(STR(?t), "entity/") AS ?t_id) ?t_label
  ?tau_start
  ?l_h_lat ?l_h_lon
  ?l_t_lat ?l_t_lon
WHERE {
  {
    { ?h wdt:P31 wd:Q198 . }    UNION
    { ?h wdt:P31 wd:Q178561 . } UNION
    { ?h wdt:P31 wd:Q350604 . } UNION
    { ?h wdt:P31 wd:Q645883 . }
  }
  BIND("occurredAt" AS ?r)
  ?h wdt:P17 wd:Q881 .
  ?h wdt:P276 ?t .
  ?h rdfs:label ?h_label . FILTER(LANG(?h_label) = "vi")
  OPTIONAL { ?t rdfs:label ?t_label . FILTER(LANG(?t_label) = "vi") }
  OPTIONAL { ?h wdt:P585 ?d1 . BIND(YEAR(?d1) AS ?ts1) }
  OPTIONAL { ?h wdt:P580 ?d2 . BIND(YEAR(?d2) AS ?ts2) }
  BIND(COALESCE(?ts1, ?ts2) AS ?tau_start)
  OPTIONAL {
    ?h p:P625/psv:P625 ?hn .
    ?hn wikibase:geoLatitude ?l_h_lat ; wikibase:geoLongitude ?l_h_lon .
  }
  OPTIONAL {
    ?t p:P625/psv:P625 ?tn .
    ?tn wikibase:geoLatitude ?l_t_lat ; wikibase:geoLongitude ?l_t_lon .
  }
}
LIMIT 500
"""

# Tên cột chuẩn cho tất cả file CSV
COLS_6TUPLE = [
    "h", "h_label", "r", "t", "t_label",
    "tau_start", "l_h_lat", "l_h_lon", "l_t_lat", "l_t_lon",
]


class WikidataCollector:
    """Thu thập dữ liệu từ Wikidata cho knowledge graph tiếng Việt."""

    def __init__(self):
        self.sparql = SPARQLWrapper(WIKIDATA_SPARQL_ENDPOINT)
        self.sparql.setReturnFormat(JSON)

    def query(self, sparql_query: str) -> list[dict]:
        """Thực thi SPARQL query và trả về danh sách bindings thô."""
        self.sparql.setQuery(sparql_query)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                results = self.sparql.query().convert()
                bindings = results["results"]["bindings"]
                logger.info(f"Query trả về {len(bindings)} kết quả.")
                return bindings
            except Exception as e:
                logger.warning(f"Lần thử {attempt}/{MAX_RETRIES} thất bại: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(REQUEST_DELAY * attempt)
        logger.error("Query thất bại sau tất cả các lần thử.")
        return []

    def collect_6tuple(self, sparql_query: str) -> list[dict]:
        """
        Thực thi SPARQL query và trả về list dict phẳng ở định dạng 6-ngôi.
        Dùng với các query SPARQL_HISTORICAL_FIGURES / _LANDMARKS / _HISTORICAL_EVENTS.
        """
        bindings = self.query(sparql_query)

        def val(binding, key):
            return binding.get(key, {}).get("value", "") or ""

        rows = []
        for b in bindings:
            rows.append({
                "h":          val(b, "h_id"),
                "h_label":    val(b, "h_label"),
                "r":          val(b, "r"),
                "t":          val(b, "t_id"),
                "t_label":    val(b, "t_label"),
                "tau_start":  val(b, "tau_start"),
                "l_h_lat":    val(b, "l_h_lat"),
                "l_h_lon":    val(b, "l_h_lon"),
                "l_t_lat":    val(b, "l_t_lat"),
                "l_t_lon":    val(b, "l_t_lon"),
            })
        return rows

    def save_6tuple_csv(self, rows: list[dict], filename: str) -> str:
        """
        Lưu danh sách 6-ngôi ra CSV trong data/raw/.

        Returns:
            Đường dẫn tuyệt đối của file đã lưu, hoặc "" nếu không có dữ liệu.
        """
        if not rows:
            logger.warning(f"Không có dữ liệu để lưu vào {filename}.")
            return ""

        os.makedirs(RAW_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DIR, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLS_6TUPLE)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Đã lưu {len(rows)} dòng vào {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Phương thức tiện ích (general-purpose)
    # ------------------------------------------------------------------

    def get_vietnamese_entities(self, entity_type: str, limit: int = 100) -> list[dict]:
        """Lấy các thực thể Wikidata có nhãn tiếng Việt theo loại QID."""
        query = f"""
        SELECT ?item ?itemLabel ?itemDescription WHERE {{
            ?item wdt:P31 wd:{entity_type} .
            ?item rdfs:label ?itemLabel .
            FILTER(LANG(?itemLabel) = "vi")
            OPTIONAL {{ ?item schema:description ?itemDescription .
                        FILTER(LANG(?itemDescription) = "vi") }}
        }}
        LIMIT {limit}
        """
        return self.query(query)

    def get_entity_by_qid(self, qid: str) -> dict:
        """Lấy thông tin chi tiết của một thực thể theo QID."""
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "languages": "vi|en",
            "format": "json",
        }
        try:
            resp = requests.get("https://www.wikidata.org/w/api.php", params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("entities", {}).get(qid, {})
        except requests.RequestException as e:
            logger.error(f"Không thể lấy thực thể {qid}: {e}")
            return {}

    def save_to_csv(self, rows: list[dict], filename: str) -> str:
        """Lưu kết quả SPARQL thô (bindings) ra CSV."""
        if not rows:
            logger.warning("Không có dữ liệu để lưu.")
            return ""

        os.makedirs(RAW_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DIR, filename)
        fieldnames = list(rows[0].keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                flat = {k: v.get("value", "") for k, v in row.items()}
                writer.writerow(flat)

        logger.info(f"Đã lưu {len(rows)} dòng vào {filepath}")
        return filepath


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    collector = WikidataCollector()
    all_rows: list[dict] = []

    print("-" * 50)
    print("1/3  Thu thap nhan vat lich su Viet Nam...")
    figures = collector.collect_6tuple(SPARQL_HISTORICAL_FIGURES)
    path = collector.save_6tuple_csv(figures, "historical_figures.csv")
    print(f"     -> {len(figures)} facts")
    all_rows.extend(figures)
    time.sleep(REQUEST_DELAY)

    print("-" * 50)
    print("2/3  Thu thap di tich & cong trinh...")
    landmarks = collector.collect_6tuple(SPARQL_LANDMARKS)
    path = collector.save_6tuple_csv(landmarks, "landmarks.csv")
    print(f"     -> {len(landmarks)} facts")
    all_rows.extend(landmarks)
    time.sleep(REQUEST_DELAY)

    print("-" * 50)
    print("3/3  Thu thap su kien lich su...")
    events = collector.collect_6tuple(SPARQL_HISTORICAL_EVENTS)
    path = collector.save_6tuple_csv(events, "historical_events.csv")
    print(f"     -> {len(events)} facts")
    all_rows.extend(events)

    print("-" * 50)
    print(f"Tong hop {len(all_rows)} facts -> combined_raw.csv ...")
    path = collector.save_6tuple_csv(all_rows, "combined_raw.csv")
    print(f"Xong! [{path}]")
