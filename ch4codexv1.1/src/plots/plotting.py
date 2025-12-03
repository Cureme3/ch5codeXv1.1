"""统一的画图入口，确保中文字体与单位符合出版标准。"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import numpy as np
import platform

# --- 1. 出版级配色 (Science/Nature 风格) ---
DEFAULT_COLORS = {
    "nominal": "#00468B",  # 深蓝 (Navy)
    "fault": "#ED0000",    # 鲜红 (Red)
    "replan": "#42B540",   # 鲜绿 (Green)
    "ref": "#925E9F",      # 紫色
    "gray": "#ADB6B6",     # 灰色
}

LINE_WIDTH = 2.0         # 核心：加粗线条，防止缩印看不清
FIG_SIZE = (6, 4.5)      # 核心：稍微加高，留出标签空间
LEGEND_FONTSIZE = 12     # 核心：图例字号

def setup_matplotlib() -> None:
    """配置 Matplotlib 以符合中文期刊要求 (宋体 + Times New Roman)。"""

    # 自动检测系统字体，确保 Windows/Linux 都能显示中文
    system_name = platform.system()
    if system_name == "Windows":
        # Windows: 优先宋体(SimSun)，其次黑体(SimHei)
        font_serif = ["SimSun", "Times New Roman", "SimHei"]
        font_sans = ["SimHei", "Arial", "Microsoft YaHei"]
    else:
        # Linux/Mac: 优先开源中文字体
        font_serif = ["Noto Serif CJK SC", "STSong", "Times New Roman"]
        font_sans = ["Noto Sans CJK SC", "STHeiti", "Arial"]

    config = {
        # --- 字体核心设置 ---
        "font.family": "serif",              # 强制衬线体 (这是论文的关键特征)
        "font.serif": font_serif + list(rcParams["font.serif"]),
        "font.sans-serif": font_sans + list(rcParams["font.sans-serif"]),

        # --- 数学公式设置 ---
        "mathtext.fontset": "stix",          # 使用 stix 字体渲染公式

        # --- 布局与线条 ---
        "axes.unicode_minus": False,         # 解决负号显示
        "axes.grid": True,                   # 开启网格
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "lines.linewidth": LINE_WIDTH,       # 全局线宽
        "axes.linewidth": 1.5,               # 坐标轴边框加粗

        # --- 字号设置 (A4纸排版标准) ---
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,                # 坐标轴标签字号
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": LEGEND_FONTSIZE,
        "figure.figsize": FIG_SIZE,

        # --- 输出设置 ---
        "figure.dpi": 300,                   # 300dpi 印刷标准
        "savefig.dpi": 300,
        "savefig.bbox": "tight",             # 防止边缘裁剪
        "savefig.pad_inches": 0.1,
    }
    rcParams.update(config)
    print(f"Plotting Style: Configured for {system_name} with serif fonts (宋体 + Times New Roman)")

# 兼容旧接口
def set_cn_pub_style() -> None:
    """配置中文期刊出版级绘图样式（兼容旧接口）。"""
    setup_matplotlib()

def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def save_figure(fig: plt.Figure, outfile: Path, dpi: int = 300) -> None:
    ensure_dir(outfile)
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi)
    fig.savefig(outfile.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)

# 保留旧接口兼容性
plot_time_series = None
plot_box = None
plot_fault_time_series = None
