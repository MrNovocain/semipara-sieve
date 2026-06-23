from __future__ import annotations

from typing import Any, Callable, TypeVar

DGP_REGISTRY: dict[str, type] = {}
METHOD_REGISTRY: dict[str, type] = {}
WEIGHT_REGISTRY: dict[str, type] = {}
BASIS_REGISTRY: dict[str, type] = {}

T = TypeVar("T", bound=type)


def _register(registry: dict[str, type], name: str) -> Callable[[T], T]:
    def wrapper(cls: T) -> T:
        if name in registry:
            raise ValueError(f"Duplicate registry name: {name}")
        registry[name] = cls
        return cls
    return wrapper


def register_dgp(name: str) -> Callable[[T], T]:
    return _register(DGP_REGISTRY, name)


def register_method(name: str) -> Callable[[T], T]:
    return _register(METHOD_REGISTRY, name)


def register_weight(name: str) -> Callable[[T], T]:
    return _register(WEIGHT_REGISTRY, name)


def register_basis(name: str) -> Callable[[T], T]:
    return _register(BASIS_REGISTRY, name)


def get_registered(registry: dict[str, type], name: str, kind: str) -> type:
    try:
        return registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "<none>"
        raise KeyError(f"Unknown {kind} '{name}'. Available: {available}") from exc


def build_weight(config: dict[str, Any]):
    cls = get_registered(WEIGHT_REGISTRY, config["name"], "weight")
    return cls(**config.get("params", {}))
