"""Name -> implementation registries for datasets, models and metrics.

Adding a new component means writing a class/function, decorating it with
``@<REGISTRY>.register("name")`` and referring to ``"name"`` from a YAML config.
Core code (CLI, pipeline, run management) never changes. See docs/extending.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A minimal string-keyed registry with a decorator-based ``register``."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under ``name``."""

        def decorator(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        """Look up a registered object, listing the available names on failure."""
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {available}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


DATASETS: Registry = Registry("dataset")
MODELS: Registry = Registry("model")
METRICS: Registry = Registry("metric")
