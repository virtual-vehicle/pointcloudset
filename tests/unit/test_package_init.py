import importlib.util
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType

import pytest_check as check


def _load_init_as_temp_package(package_name: str):
    init_path = Path(__file__).parents[2] / "src" / "pointcloudset" / "__init__.py"

    dataset_mod = ModuleType(f"{package_name}.dataset")
    pointcloud_mod = ModuleType(f"{package_name}.pointcloud")

    class DummyDataset:
        pass

    class DummyPointCloud:
        pass

    dataset_mod.Dataset = DummyDataset
    pointcloud_mod.PointCloud = DummyPointCloud

    sys.modules[f"{package_name}.dataset"] = dataset_mod
    sys.modules[f"{package_name}.pointcloud"] = pointcloud_mod

    spec = importlib.util.spec_from_file_location(
        package_name,
        str(init_path),
        submodule_search_locations=[str(init_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module, DummyDataset, DummyPointCloud


def _cleanup_temp_package(package_name: str) -> None:
    keys = [
        package_name,
        f"{package_name}.dataset",
        f"{package_name}.pointcloud",
    ]
    for key in keys:
        sys.modules.pop(key, None)


def test_init_version_loaded_from_metadata(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _: "9.9.9")

    package_name = "pointcloudset_init_success"
    try:
        module, DummyDataset, DummyPointCloud = _load_init_as_temp_package(package_name)

        check.equal(module.__version__, "9.9.9")
        check.equal(module.Dataset, DummyDataset)
        check.equal(module.PointCloud, DummyPointCloud)
    finally:
        _cleanup_temp_package(package_name)


def test_init_version_fallback_when_package_not_found(monkeypatch):
    def raise_not_found(_):
        raise PackageNotFoundError

    monkeypatch.setattr("importlib.metadata.version", raise_not_found)

    package_name = "pointcloudset_init_fallback"
    try:
        module, DummyDataset, DummyPointCloud = _load_init_as_temp_package(package_name)

        check.equal(module.__version__, "0.0.0")
        check.equal(module.Dataset, DummyDataset)
        check.equal(module.PointCloud, DummyPointCloud)
    finally:
        _cleanup_temp_package(package_name)
