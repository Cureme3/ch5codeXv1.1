"\"\"\"阶段事件时间轴工具。\"\"\""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class Timeline:
    liftoff_s: float
    stage1_sep_s: float
    stage2_sep_s: float
    stage3_sep_s: float
    fairing_jettison_s: float
    orbit_insertion_s: float


def load_timeline(config_path: str | Path) -> Timeline:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    tl: Dict = cfg["timeline"]
    return Timeline(
        liftoff_s=tl["liftoff_s"],
        stage1_sep_s=tl["stage1_sep_s"],
        stage2_sep_s=tl["stage2_sep_s"],
        stage3_sep_s=tl.get("stage3_sep_s", tl["stage3_burnout_s"]),
        fairing_jettison_s=tl["fairing_jettison_s"],
        orbit_insertion_s=tl["orbit_insertion_s"],
    )
