import inspect
import importlib
import pytest

import collective_encoder
from collective_encoder.common.module import CEModule
from collective_encoder.common import registry


EXPECTED_KINDS = {
    "dataanalysers",
    "datalabelers",
    "datamodules",
    "datareaders",
    "datasets",
    "losses",
    "metrics",
    "models",
    "testplotters",
}


KNOWN_BROKEN_PLUGINS = {
    ("datasets", "COLVAR"): (
        "Bug in collective_encoder/datasets/colvar.py: ColvarDataset does not implement "
        "abstract method BaseDataset.get_data()"
    ),
}

IGNORED_DEV_PLUGINS = {
    ("models", "BGE_V2"),
}


def _all_registered_pairs():
    pairs = []
    for kind in registry.kinds():
        for name in registry.available(kind):
            if (kind, name) in IGNORED_DEV_PLUGINS:
                continue
            reason = KNOWN_BROKEN_PLUGINS.get((kind, name))
            if reason is not None:
                pairs.append(
                    pytest.param(
                        kind,
                        name,
                        marks=pytest.mark.xfail(strict=True, reason=reason),
                    )
                )
            else:
                pairs.append((kind, name))
    return pairs


@pytest.mark.unit
def test_expected_kinds_registered():
    """Verify all core plugin kinds are registered when collective_encoder is imported."""
    registered_kinds = set(registry.kinds())
    assert EXPECTED_KINDS.issubset(registered_kinds)


@pytest.mark.unit
@pytest.mark.parametrize("kind,name", _all_registered_pairs())
def test_registered_plugin_resolves_and_honors_contracts(kind: str, name: str):
    """
    Every registered plugin across all kinds must:
    1. Resolve via `get_module(kind, name)` without ImportError or TypeError.
    2. Be a concrete (non-abstract) class inheriting from `_BASES[kind]` and `CEModule`.
    3. Define valid `_REQUIRED_ARGS` (list) and `_OPTIONAL_ARGS` (dict) without overlap.
    4. Be lazily exported on its parent `collective_encoder.<kind>` subpackage.
    """
    cls = registry.get_module(kind, name)
    base_cls = registry._BASES[kind]

    assert inspect.isclass(cls), f"{kind}/{name} did not resolve to a class"
    assert issubclass(cls, base_cls), f"{cls.__name__} does not subclass {base_cls.__name__}"
    assert issubclass(cls, CEModule), f"{cls.__name__} does not subclass CEModule"
    assert not inspect.isabstract(cls), (
        f"{kind}/{name} ({cls.__name__}) is still abstract; "
        f"unimplemented methods: {getattr(cls, '__abstractmethods__', set())}"
    )

    assert isinstance(cls._REQUIRED_ARGS, list), f"{cls.__name__}._REQUIRED_ARGS must be a list"
    assert isinstance(cls._OPTIONAL_ARGS, dict), f"{cls.__name__}._OPTIONAL_ARGS must be a dict"

    overlap = set(cls._REQUIRED_ARGS) & set(cls._OPTIONAL_ARGS.keys())
    assert not overlap, (
        f"{cls.__name__} has overlapping keys in _REQUIRED_ARGS and _OPTIONAL_ARGS: {overlap}"
    )

    # Verify lazy export on collective_encoder.<kind> if it belongs to collective_encoder
    if cls.__module__.startswith(f"collective_encoder.{kind}"):
        subpkg = importlib.import_module(f"collective_encoder.{kind}")
        assert cls.__name__ in dir(subpkg)
        assert getattr(subpkg, cls.__name__) is cls


@pytest.mark.unit
def test_registry_custom_plugin_and_filepath_loading(tmp_path):
    """
    Test defining a custom kind, registering from both module path and a `.py` file path,
    duplicate protection (`overwrite=False` vs `overwrite=True`), and base-class checking.
    """
    class CustomBase(CEModule):
        pass

    kind = "_test_custom_kind"
    registry.define_kind(kind, CustomBase)
    assert kind in registry.kinds()

    # Write a standalone plugin file to tmp_path
    plugin_file = tmp_path / "my_custom_plugin.py"
    plugin_file.write_text(
        "from collective_encoder.common.module import CEModule\n"
        f"from collective_encoder.common import registry\n"
        f"Base = registry._BASES[{kind!r}]\n"
        "class FilePlugin(Base):\n"
        "    _IDENTIFIER = 'FILE_PLUGIN'\n"
        "class InvalidPlugin:\n"
        "    pass\n"
    )

    registry.register(kind, "FROM_FILE", str(plugin_file), "FilePlugin")
    assert "FROM_FILE" in registry.available(kind)

    resolved = registry.get_module(kind, "FROM_FILE")
    assert resolved.__name__ == "FilePlugin"
    assert issubclass(resolved, CustomBase)

    # Duplicate registration without overwrite=True must raise ValueError
    with pytest.raises(ValueError, match="already registered"):
        registry.register(kind, "FROM_FILE", str(plugin_file), "FilePlugin", overwrite=False)

    # With overwrite=True it succeeds
    registry.register(kind, "FROM_FILE", str(plugin_file), "FilePlugin", overwrite=True)

    # Registering a class that does not inherit from CustomBase must raise TypeError on get_module
    registry.register(kind, "INVALID", str(plugin_file), "InvalidPlugin")
    with pytest.raises(TypeError, match="does not subclass"):
        registry.get_module(kind, "INVALID")

    # Unknown plugin and unknown kind raise KeyError
    with pytest.raises(KeyError, match="Unknown"):
        registry.get_module(kind, "NON_EXISTENT_PLUGIN")

    with pytest.raises(KeyError, match="Unknown plugin kind"):
        registry.get_module("non_existent_kind", "ANY")

    # Clean up test kind from global registry
    registry._REGISTRY.pop(kind, None)
    registry._BASES.pop(kind, None)
