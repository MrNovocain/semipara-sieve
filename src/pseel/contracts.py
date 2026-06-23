from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Dataset:
    y: np.ndarray
    x_lag: np.ndarray
    w_lag: np.ndarray
    u: np.ndarray
    m_w: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.y)
        arrays = {
            "x_lag": self.x_lag,
            "w_lag": self.w_lag,
            "u": self.u,
            "m_w": self.m_w,
        }
        for name, value in arrays.items():
            if len(value) != n:
                raise ValueError(f"Dataset field {name} has length {len(value)}; expected {n}.")
        for name, value in {"y": self.y, **arrays}.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Dataset field {name} contains non-finite values.")

    @property
    def T(self) -> int:
        return int(len(self.y))


@dataclass(frozen=True)
class ELResult:
    statistic: float
    lambda_hat: float | None
    feasible: bool
    message: str = "ok"


@dataclass(frozen=True)
class MethodResult:
    method_name: str
    el_stat: float
    reject_5: bool
    beta0: float
    feasible: bool
    lambda_hat: float | None
    diagnostics: dict[str, float] = field(default_factory=dict)
