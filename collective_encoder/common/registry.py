# mylib/registry.py
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_MISSING = object()


@dataclass
class _Entry:
    source: str          # "/abs/path/file.py" or "package.module"
    attr: str            # class/function name inside that module
    obj: Any = _MISSING  # cached after first load


# {kind: {name: _Entry}}
_REGISTRY: dict[str, dict[str, _Entry]] = {}

# {kind: base class (or tuple of classes) that plugins must subclass, or None}
_BASES: dict[str, type | tuple[type, ...] | None] = {}


def register(kind: str, name: str, source: str | Path, attr: str | None = None,
             *, overwrite: bool = False) -> None:
    if kind not in _BASES:
        raise KeyError(f"Unknown plugin kind {kind!r}. Call define_kind() first. "
                       f"Defined kinds: {sorted(_BASES)}")
    if attr is None:
        source, attr = source

    entries = _REGISTRY[kind]
    if name in entries and not overwrite:
        raise ValueError(f"{kind}/{name!r} is already registered "
                         f"(from {entries[name].source}); use overwrite=True")
    entries[name] = _Entry(source, attr)


def get(kind: str, name: str) -> Any:
    if kind not in _BASES:
        raise KeyError(f"Unknown plugin kind {kind!r}. Kinds: {sorted(_BASES)}")
    try:
        entry = _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"Unknown {kind} plugin {name!r}. "
                       f"Available: {sorted(_REGISTRY[kind])}") from None

    if entry.obj is _MISSING:
        module = _import_source(entry.source)
        try:
            obj = getattr(module, entry.attr)
        except AttributeError:
            raise ImportError(f"{entry.source} has no attribute {entry.attr!r}") from None

        _check_base(kind, name, entry, obj)   # raises before caching
        entry.obj = obj
    return entry.obj

get_module = get  # alias

def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


def kinds() -> list[str]:
    return sorted(_REGISTRY)


# ---------- helpers ----------


def _is_file_path(source: str) -> bool:
    return source.endswith(".py") or "/" in source or "\\" in source


def _import_source(source: str) -> ModuleType:
    if not _is_file_path(source):
        return importlib.import_module(source)

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Plugin file not found: {path}")

    # unique, stable module name so two files named "utils.py" don't collide
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:8]
    mod_name = f"_ce_plugin_{digest}_{path.stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module          # needed before exec for dataclasses, pickle, etc.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    return module


def define_kind(kind: str, base: type | tuple[type, ...] | None = None) -> None:
    """Declare a plugin kind and the base class its plugins must inherit from."""
    if kind in _BASES and _BASES[kind] is not base:
        raise ValueError(f"Kind {kind!r} is already defined with base {_BASES[kind]!r}")
    _BASES[kind] = base
    _REGISTRY.setdefault(kind, {})

def _check_base(kind: str, name: str, entry: _Entry, obj: Any) -> None:
    base = _BASES[kind]
    if base is None:
        return
    if not isinstance(obj, type) or not issubclass(obj, base):
        expected = (base.__name__ if isinstance(base, type)
                    else " | ".join(b.__name__ for b in base))
        actual = obj.__name__ if isinstance(obj, type) else type(obj).__name__
        raise TypeError(
            f"{kind}/{name!r} ({entry.source}:{entry.attr}) is {actual!r}, "
            f"which does not subclass {expected}"
        )

def lazy_exports(package: str, targets: dict[str, str]):
    """
    targets: {"CsvLoader": "mylib.dataloaders.csv:CsvLoader", ...}
    Returns (__getattr__, __dir__) to assign in the package's __init__.py.
    """
    def __getattr__(name: str):
        try:
            module_path, _, attr = targets[name].partition(":")
        except KeyError:
            raise AttributeError(f"module {package!r} has no attribute {name!r}") from None
        obj = getattr(importlib.import_module(module_path), attr)
        sys.modules[package].__dict__[name] = obj      # cache: next access skips __getattr__
        return obj

    def __dir__():
        return sorted(set(sys.modules[package].__dict__) | set(targets))

    return __getattr__, __dir__