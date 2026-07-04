"""
Loader chung cho config.yaml — dùng bởi mọi script trong src/ (data/, model/,
train/, eval/, baselines/, viz/) theo hợp đồng: mỗi script nhận --config
(mặc định config.yaml ở gốc repo) cộng theo các override dạng "key.sub=value".

Ví dụ:
    python -m src.train.run --config config.yaml --set optimizer.batch_size=16
"""

import argparse
import os
from typing import Any

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_nested(cfg: dict, dotted_key: str, value: str) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    leaf = keys[-1]
    old = node.get(leaf)
    if isinstance(old, bool):
        node[leaf] = value.lower() in ("1", "true", "yes")
    elif isinstance(old, int):
        node[leaf] = int(value)
    elif isinstance(old, float):
        node[leaf] = float(value)
    else:
        node[leaf] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """overrides: danh sách "key.sub=value" (vd ["train.batch_size=16"])."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override không hợp lệ (thiếu '='): {item}")
        key, value = item.split("=", 1)
        _set_nested(cfg, key.strip(), value.strip())
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Đường dẫn config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                         help='Override 1 giá trị, vd --set optimizer.batch_size=16 (có thể lặp lại)')
    return parser


def load_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    return apply_overrides(cfg, args.overrides)
