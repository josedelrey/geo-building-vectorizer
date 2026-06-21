from collections.abc import Callable
from typing import Any

from geobuild.vectorize.base import Vectorizer


VectorizerFactory = Callable[[dict[str, Any]], Vectorizer]
_VECTORIZERS: dict[str, VectorizerFactory] = {}


def register_vectorizer(
    name: str,
    factory: VectorizerFactory,
) -> None:
    normalized_name = _normalize_name(name)

    if normalized_name in _VECTORIZERS:
        raise ValueError(f"Vectorizer is already registered: {normalized_name!r}")

    _VECTORIZERS[normalized_name] = factory


def build_vectorizer(config: dict[str, Any]) -> Vectorizer:
    _register_builtin_vectorizers()
    vectorizer_config = config.get("vectorizer", config)
    name = _normalize_name(vectorizer_config.get("name"))

    if name not in _VECTORIZERS:
        available = sorted(_VECTORIZERS)
        raise ValueError(
            f"Unknown vectorizer {name!r}. "
            f"Available vectorizers: {available if available else 'none registered'}"
        )

    vectorizer = _VECTORIZERS[name](dict(vectorizer_config))

    if not isinstance(vectorizer, Vectorizer):
        raise TypeError(
            f"Vectorizer factory for {name!r} returned "
            f"{type(vectorizer).__name__}, expected Vectorizer"
        )

    return vectorizer


def _normalize_name(name: Any) -> str:
    if name is None:
        raise KeyError("Vectorizer config must include a name")

    normalized = str(name).strip().lower()

    if not normalized:
        raise ValueError("Vectorizer name cannot be empty")

    return normalized


def _register_builtin_vectorizers() -> None:
    import geobuild.vectorize.mask_cc  # noqa: F401
