import pytest
import importlib
import pkgutil
import inspect

from collective_encoder.datalabelers.base import BaseLabeler

def get_subclasses(module_name, base_class):
    module = importlib.import_module(module_name)
    subclasses = []
    if hasattr(module, "__path__"):
        for loader, name, is_pkg in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
            try:
                submodule = importlib.import_module(name)
                for item_name, item in inspect.getmembers(submodule):
                    if inspect.isclass(item) and issubclass(item, base_class) and item is not base_class and item.__module__ == name:
                        subclasses.append(item)
            except Exception:
                pass
    return list(set(subclasses))

@pytest.mark.parametrize("cls", get_subclasses("collective_encoder.datalabelers", BaseLabeler))
def test_datalabeler_interface(cls):
    assert issubclass(cls, BaseLabeler)
    if not inspect.isabstract(cls):
        try:
            assert hasattr(cls, "get_label_names")
            assert hasattr(cls, "compute")
        except TypeError as e:
            pytest.fail(f"Could not instantiate {cls.__name__} due to unimplemented abstract methods: {e}")
