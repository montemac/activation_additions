"""
Smoke integration test for `sparse_coding.py`.

Note that this integration test will necessarily be somewhat slow.
"""


import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def mock_acts_config():
    """Mock `act_config.yaml`."""
    smoke_config: dict = {
        "MODEL_DIR": "google/bert_uncased_L-2_H-128_A-2",
        "ACTS_LAYER": 1,
        "SMALL_MODEL_MODE": True,
        "PROJECTION_FACTOR": 1,
        "PROMPT_IDS_PATH": "acts_data/mini_test_activations_prompt_ids.npy",
        "ACTS_DATA_PATH": "acts_data/mini_test_activations_dataset.pt",
        "ENCODER_PATH": "acts_data/mini_test_learned_encoder.pt",
        "BIASES_PATH": "acts_data/mini_test_learned_biases.pt",
        "TOP_K_INFO_PATH": "acts_data/mini_test_token_info.csv",
        "LAMBDA_L1": 1,
        "LEARNING_RATE": 1.0e-2,
        "NUM_WORKERS": 0,
        "NUM_QUESTIONS_EVALED": 5,
        "EPOCHS": 1,
    }
    smoke_config_path = Path("../sparse_coding/act_config.yaml")
    with open(smoke_config_path, "w", encoding="utf-8") as f:
        yaml.dump(smoke_config, f)


@pytest.mark.usefixtures("mock_acts_config")
def test_smoke_sparse_coding():
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
