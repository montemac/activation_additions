"""
Smoke integration test for `sparse_coding.py`.

Note that this integration test will necessarily be somewhat slow.
"""


import runpy

import pytest
import yaml


@pytest.fixture
def mock_load_yaml_constants(monkeypatch):
    """Load from the smoke test configuration YAML files."""

    def mock_load():
        """Load config files with get() methods."""

        try:
            with open("smoke_test_access.yaml", "r", encoding="utf-8") as f:
                access = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
        with open("smoke_test_config.yaml", "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)

        return access, config

    monkeypatch.setattr(
        "sparse_coding.utils.configure.load_yaml_constants", mock_load
    )


def test_smoke_sparse_coding(
    mock_load_yaml_constants,
):  # pylint: disable=redefined-outer-name, unused-argument
    """Run the submodule scripts in sequence."""
    for script in [
        "acts_collect",
        "autoencoder",
        "feature_tokens",
    ]:
        try:
            print(f"Starting smoke test for {script}...")
            runpy.run_module(f"sparse_coding.{script}")
            print(f"Smoke test for {script} passed!")
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail(f"Smoke test for {script} failed: {e}")
