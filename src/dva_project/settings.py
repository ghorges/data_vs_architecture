from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
EXTERNAL_DIR = PROJECT_ROOT / "external"

MANIFEST_DIR = RAW_DIR / "manifests"
MODEL_YAML_DIR = RAW_DIR / "model_yamls"
WBM_DIR = RAW_DIR / "wbm"
DISCOVERY_PREDICTIONS_DIR = RAW_DIR / "predictions" / "discovery"

MATBENCH_DISCOVERY_REPO_URL = "https://github.com/janosh/matbench-discovery.git"
MATBENCH_DISCOVERY_REPO_DIR = EXTERNAL_DIR / "matbench-discovery"
MATBENCH_DISCOVERY_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/janosh/matbench-discovery/main"
)
FIGSHARE_DISCOVERY_ARTICLE_ID = 28187990
FIGSHARE_DISCOVERY_FILES_API = (
    f"https://api.figshare.com/v2/articles/{FIGSHARE_DISCOVERY_ARTICLE_ID}/files"
)

WBM_SUMMARY_FILE_NAME = "2023-12-13-wbm-summary.csv.gz"
DATASETS_YAML_NAME = "datasets.yml"
