"""
Geocoding qua OpenStreetMap Nominatim.
Miễn phí, không cần API key.
Rate limit: 1 request/giây theo chính sách Nominatim.
"""

import time
import logging
import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OSM_DELAY, TIMEOUT

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


class OSMResolver:
    """Geocoder dùng Nominatim (OpenStreetMap)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "stkg_vietnamese/1.0 (research; contact: stkg@example.com)",
            "Accept-Language": "vi,en",
        })
        self._last_call = 0.0

    def _wait(self):
        """Đảm bảo đúng rate limit 1 req/giây."""
        elapsed = time.time() - self._last_call
        if elapsed < OSM_DELAY:
            time.sleep(OSM_DELAY - elapsed)
        self._last_call = time.time()

    def resolve(self, place_name: str, country: str = "Vietnam") -> tuple[float, float] | None:
        """
        Geocode tên địa danh thành (lat, lon).

        Args:
            place_name: Tên địa danh tiếng Việt
            country: Tên quốc gia để thu hẹp kết quả

        Returns:
            (lat, lon) hoặc None nếu không tìm được
        """
        if not place_name or not place_name.strip():
            return None

        self._wait()
        params = {
            "q": f"{place_name}, {country}",
            "format": "json",
            "limit": 1,
            "accept-language": "vi",
        }
        try:
            resp = self.session.get(NOMINATIM_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            results = resp.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                logger.debug(f"OSM resolved '{place_name}' -> ({lat:.4f}, {lon:.4f})")
                return lat, lon
            logger.debug(f"OSM: khong tim thay '{place_name}'")
            return None
        except Exception as e:
            logger.warning(f"OSM error cho '{place_name}': {e}")
            return None

    def resolve_batch(self, place_names: list[str]) -> dict[str, tuple[float, float] | None]:
        """
        Geocode nhiều địa danh, trả về dict {tên: (lat, lon) | None}.
        Tự động bỏ qua trùng lặp.
        """
        results = {}
        unique = list(dict.fromkeys(p for p in place_names if p))
        for i, name in enumerate(unique, 1):
            coords = self.resolve(name)
            results[name] = coords
            if i % 10 == 0:
                logger.info(f"OSM: {i}/{len(unique)} ({sum(v is not None for v in results.values())} hits)")
        return results


if __name__ == "__main__":
    resolver = OSMResolver()
    test_places = ["Hà Nội", "Thành phố Hồ Chí Minh", "Nghệ An", "Điện Biên Phủ", "Huế"]
    for place in test_places:
        coords = resolver.resolve(place)
        print(f"{place}: {coords}")
