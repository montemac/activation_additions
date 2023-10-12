"""
Smoke integration test for `sparse_coding.py`.

Note that this integration test will necessarily be somewhat slow.
"""


import subprocess

import pytest
import yaml

from sparse_coding.utils.configure import load_yaml_constants


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

    monkeypatch.setattr(load_yaml_constants, "load_yaml_constants", mock_load)


@pytest.mark.slow
def test_smoke_sparse_coding(mock_load_yaml_constants):
    """Run the submodule scripts in sequence."""
    try:
        for script in [
            "acts_collect.py",
            "autoencoder.py",
            "feature_tokens.py",
        ]:
            subprocess.run(
                ["python3", f"../sparse_coding/{script}"], check=True
            )

        print("Smoke test passed!")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Smoke test failed: {e}")
