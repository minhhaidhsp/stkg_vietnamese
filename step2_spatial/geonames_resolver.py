"""
Geocoding qua GeoNames API.
Cần đăng ký miễn phí tại https://www.geonames.org/login
Sau đó set GEONAMES_USERNAME trong config.py.
Rate limit: 1000 requests/giờ (free tier).
"""

import logging
import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEONAMES_USERNAME, TIMEOUT

logger = logging.getLogger(__name__)

GEONAMES_URL = "http://api.geonames.org/searchJSON"


class GeoNamesResolver:
    """Geocoder dùng GeoNames API."""

    def __init__(self, username: str = ""):
        self.username = username or GEONAMES_USERNAME
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "stkg_vietnamese/1.0"})

    @property
    def available(self) -> bool:
        return bool(self.username)

    def resolve(self, place_name: str, country: str = "VN") -> tuple[float, float] | None:
        """
        Geocode tên địa danh thành (lat, lon).

        Args:
            place_name: Tên địa danh
            country: ISO-2 quốc gia ('VN' cho Việt Nam)

        Returns:
            (lat, lon) hoặc None nếu không tìm được / chưa cấu hình username
        """
        if not self.available:
            logger.debug("GeoNames: username chưa cấu hình, bỏ qua.")
            return None
        if not place_name or not place_name.strip():
            return None

        params = {
            "q": place_name,
            "country": country,
            "maxRows": 1,
            "username": self.username,
            "lang": "vi",
            "orderby": "relevance",
        }
        try:
            resp = self.session.get(GEONAMES_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            if "status" in data:
                logger.warning(f"GeoNames error: {data['status'].get('message')}")
                return None

            geonames = data.get("geonames", [])
            if geonames:
                g = geonames[0]
                lat, lon = float(g["lat"]), float(g["lng"])
                logger.debug(f"GeoNames resolved '{place_name}' -> ({lat:.4f}, {lon:.4f})")
                return lat, lon

            logger.debug(f"GeoNames: khong tim thay '{place_name}'")
            return None
        except Exception as e:
            logger.warning(f"GeoNames error cho '{place_name}': {e}")
            return None

    def resolve_batch(self, place_names: list[str]) -> dict[str, tuple[float, float] | None]:
        """Geocode nhiều địa danh. Trả về dict {tên: (lat, lon) | None}."""
        if not self.available:
            return {name: None for name in place_names}
        results = {}
        unique = list(dict.fromkeys(p for p in place_names if p))
        for name in unique:
            results[name] = self.resolve(name)
        return results


if __name__ == "__main__":
    resolver = GeoNamesResolver()
    if not resolver.available:
        print("Chua cau hinh GEONAMES_USERNAME trong config.py")
        print("Dang ky mien phi tai: https://www.geonames.org/login")
    else:
        for place in ["Ha Noi", "Ho Chi Minh City", "Nghe An"]:
            print(f"{place}: {resolver.resolve(place)}")
