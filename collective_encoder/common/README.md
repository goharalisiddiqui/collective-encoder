# Common Utilities & Registry

This module provides the foundational building blocks shared across all subpackages in `collective_encoder`:

1. **`module.py` (`CEModule`)**: The root base class for all configurable components, handling required/optional argument validation (`_REQUIRED_ARGS`, `_OPTIONAL_ARGS`), logging, and run directory management.
2. **`registry.py`**: The centralized, lazy-loading plugin registry that replaces per-folder `resolver.py` files.
3. **`config_check.py`**: YAML configuration validation helpers (e.g., checking for duplicate keys and required fields).

## Centralized Registry (`registry.py`)

All pluggable component families (`datamodules`, `datareaders`, `datasets`, `datalabelers`, `dataanalysers`, `losses`, `metrics`, `models`, `testplotters`) are managed through `collective_encoder.common.registry`.

### Core Functions

- **`define_kind(kind: str, base: type | tuple[type, ...] | None)`**:
  Declares a plugin category (e.g., `"datamodules"`) and the abstract base class that all registered plugins of that kind must subclass.
- **`register(kind: str, name: str, source: str | Path, attr: str | None = None, *, overwrite: bool = False)`**:
  Registers a class under `kind` and `name` without importing `source` eagerly. `source` can be either a dotted Python module path (e.g. `"collective_encoder.datamodules.coordinates"`) or a filesystem path to a `.py` file.
- **`get(kind: str, name: str)` / `get_module(kind: str, name: str)`**:
  Lazily imports the target module on first request, verifies that the resolved class inherits from the kind's base class, caches it, and returns the class.
- **`available(kind: str) -> list[str]` & `kinds() -> list[str]`**:
  Inspect registered plugin names and kinds.
- **`lazy_exports(package: str, targets: dict[str, str])`**:
  Generates module-level `__getattr__` and `__dir__` functions for a subpackage's `__init__.py` so classes are only imported when accessed.

### Standard `__init__.py` Pattern

Each subpackage registers its built-in classes in its `__init__.py`:

```python
from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import BaseDataModule

define_kind("datamodules", BaseDataModule)

_BUILTINS = {
    "COORDINATES": ("collective_encoder.datamodules.coordinates", "CoordinatesDataModule"),
    "COLVAR": ("collective_encoder.datamodules.colvars", "ColvarsDataModule"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("datamodules", _name, _source, _attr)

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["BaseDataModule"] + [t for m, t in _BUILTINS.values()]
```
