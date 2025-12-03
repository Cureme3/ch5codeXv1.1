"""SCvx / SCP 轨迹优化模块入口。

该包目前提供离散化接口、SOCP 子问题构造器以及 SCvxPlanner 框架。
详细实现将在后续步骤中逐步填充，当前文件仅用于暴露公共 API。
"""

from .discretization import (
    DiscreteDynamics,
    DiscreteTrajectory,
    GridConfig,
    NominalTrajectoryBundle,
    TimeGrid,
)
from .socp_problem import SOCPProblemBuilder, TrustRegionConfig
from .scvx import SCvxPlanner, SCvxResult

__all__ = [
    "DiscreteDynamics",
    "DiscreteTrajectory",
    "GridConfig",
    "NominalTrajectoryBundle",
    "TimeGrid",
    "SOCPProblemBuilder",
    "TrustRegionConfig",
    "SCvxPlanner",
    "SCvxResult",
]
