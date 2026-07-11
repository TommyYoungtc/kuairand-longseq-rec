"""配置加载:yaml → 简单命名空间。"""
import argparse
from pathlib import Path

import yaml


class Config(dict):
    """dict 的属性访问包装:cfg.key 等价 cfg['key']。"""

    __getattr__ = dict.__getitem__

    @property
    def raw(self) -> Path:
        return Path(self["raw_dir"])

    @property
    def out(self) -> Path:
        return Path(self["out_dir"])

    @property
    def reports(self) -> Path:
        return Path(self["reports_dir"])


def load_config(path: str | None = None) -> Config:
    overrides = []
    if path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--set", nargs="*", default=[],
                            help="覆盖配置项,如 --set rank.model=din rank.hist_len=50")
        args = parser.parse_args()
        path, overrides = args.config, args.set
    with open(path, encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    for kv in overrides:
        key, _, val = kv.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
        print(f"[config override] {key} = {node[parts[-1]]}")
    cfg["_config_path"] = str(path) + ("" if not overrides else f" ({' '.join(overrides)})")
    cfg.out.mkdir(parents=True, exist_ok=True)
    cfg.reports.mkdir(parents=True, exist_ok=True)
    return cfg
