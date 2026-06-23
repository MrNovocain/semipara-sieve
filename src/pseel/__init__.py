"""Minimal reproducible profile-sieve empirical likelihood package."""

from .contracts import Dataset, ELResult, MethodResult
from .el import empirical_likelihood_scalar

__all__ = ["Dataset", "ELResult", "MethodResult", "empirical_likelihood_scalar"]
