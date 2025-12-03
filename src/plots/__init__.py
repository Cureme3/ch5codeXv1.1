"""绘图模块 - 中文出版级样式。"""

from .plotting import (
    set_cn_pub_style,
    setup_matplotlib,
    get_label,
    save_figure,
    plot_time_series,
    plot_comparison,
    plot_fault_comparison,
    plot_box,
    CN_LABELS,
    DEFAULT_COLORS,
    LINE_WIDTH,
    FIG_SIZE,
    LEGEND_FONTSIZE,
)

__all__ = [
    "set_cn_pub_style",
    "setup_matplotlib",
    "get_label",
    "save_figure",
    "plot_time_series",
    "plot_comparison",
    "plot_fault_comparison",
    "plot_box",
    "CN_LABELS",
    "DEFAULT_COLORS",
    "LINE_WIDTH",
    "FIG_SIZE",
    "LEGEND_FONTSIZE",
]
