import os

# Đường dẫn gốc của dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Cấu hình Wikidata
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

# Cấu hình Wikipedia tiếng Việt
WIKIPEDIA_API_URL = "https://vi.wikipedia.org/w/api.php"
WIKIPEDIA_LANG = "vi"

# Cấu hình thu thập dữ liệu
REQUEST_DELAY = 1.0  # giây giữa các request
MAX_RETRIES = 3
TIMEOUT = 30  # giây

# Cấu hình lịch âm
LUNAR_CALENDAR_START_YEAR = 1900
LUNAR_CALENDAR_END_YEAR = 2100

# Cấu hình bước 3 - Visual
VISUAL_DIR  = os.path.join(DATA_DIR, "visual")
IMAGES_DIR  = os.path.join(VISUAL_DIR, "images")
VIT_MODEL   = "google/vit-base-patch16-224"
CLIP_MODEL  = "openai/clip-vit-base-patch32"
IMAGE_WIDTH = 400               # px khi tải ảnh từ Wikimedia

# Cấu hình bước 3 - Kế thừa
INHERITED_DIR = os.path.join(DATA_DIR, "inherited")  # ICEWS download vào đây
STEP3_DIR     = os.path.join(DATA_DIR, "step3")       # Output bước 3

# Cấu hình bước 2 - Spatial
SPATIAL_DIR = os.path.join(DATA_DIR, "spatial")
GEONAMES_USERNAME = ""          # Đăng ký miễn phí tại geonames.org
OSM_DELAY = 1.1                 # giây giữa các request Nominatim (rate limit: 1/s)
VIETNAM_LAT = (8.0, 23.5)      # bbox vĩ độ Việt Nam
VIETNAM_LON = (102.0, 110.0)   # bbox kinh độ Việt Nam
