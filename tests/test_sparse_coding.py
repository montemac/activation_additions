"""Unit tests for the `sparse_coding` submodule."""


from collections import defaultdict

import pytest
import torch as t
import transformers
from accelerate import Accelerator

from sparse_coding.utils.top_k import (
    per_input_token_effects,
    project_activations,
    select_top_k_tokens,
)


# Test determinism.
t.manual_seed(0)


@pytest.fixture
def mock_autoencoder():
    """Return a mock model, its tokenizer, and its accelerator."""

    class MockEncoder:
        """Mock an encoder model."""

        def __init__(self):
            """Initialize the mock encoder."""
            self.encoder_layer = t.nn.Linear(512, 1024)
            t.nn.Sequential(self.encoder_layer, t.nn.ReLU())

        def __call__(self, inputs):
            """Mock projection behavior."""
            return self.encoder_layer(inputs)

    mock_encoder = MockEncoder()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m"
    )
    accelerator = Accelerator()

    return mock_encoder, tokenizer, accelerator


@pytest.fixture
def mock_data():
    """Return mock input token ids and autoencoder activation by q tensors."""

    question_token_ids: list[list[int]] = [
        [0, 1, 2, 3, 4, 0],
        [9, 10, 11, 1, 2, 3],
    ]
    feature_activations: list[t.Tensor] = [t.randn(6, 1024) for _ in range(2)]

    return question_token_ids, feature_activations


def test_per_input_token_effects(  # pylint: disable=redefined-outer-name
    mock_autoencoder, mock_data
):
    """Test `per_input_token_effects`."""

    # Pytest fixture injections.
    mock_encoder, tokenizer, accelerator = mock_autoencoder
    question_token_ids, feature_activations = mock_data

    batch_size = 200
    large_model_mode = False

    mock_effects = per_input_token_effects(
        question_token_ids,
        feature_activations,
        mock_encoder,
        tokenizer,
        accelerator,
        batch_size,
        large_model_mode,
    )

    assert isinstance(mock_effects, defaultdict)
    assert isinstance(mock_effects[0], defaultdict)
    assert len(mock_effects) == 1024  # PROJECTION_DIM feature dimensions.
    assert len(mock_effects[0]) == 8  # Eight unique tokens.


def test_project_activations(  # pylint: disable=redefined-outer-name
    mock_autoencoder,
):
    """Test `project_activations`."""

    acts_list = [t.randn(5, 512) for _ in range(2)]
    mock_encoder, _, accelerator = mock_autoencoder

    mock_projections = project_activations(
        acts_list, mock_encoder, accelerator
    )

    assert isinstance(mock_projections, list)
    assert isinstance(mock_projections[0], t.Tensor)
    assert mock_projections[0].shape == (5, 1024)


def test_select_top_k_tokens():
    """Test `select_top_k_tokens`."""

    def inner_defaultdict():
        """Return a new inner defaultdict."""
        return defaultdict(str)

    mock_effects: defaultdict[int, defaultdict[str, float]] = defaultdict(
        inner_defaultdict
    )
    mock_effects[0]["a"] = 1.0
    mock_effects[0]["b"] = 0.5
    mock_effects[0]["c"] = 0.25
    mock_effects[0]["d"] = 0.125
    mock_effects[0]["e"] = 0.0625
    mock_effects[1]["a"] = 0.5
    mock_effects[1]["b"] = 0.25
    mock_effects[1]["c"] = 0.125
    mock_effects[1]["d"] = 0.0625
    mock_effects[1]["e"] = 0.03125

    top_k: int = 3

    mock_top_k_tokens = select_top_k_tokens(mock_effects, top_k)

    assert isinstance(mock_top_k_tokens, defaultdict)
    assert isinstance(mock_top_k_tokens[0], list)
    assert isinstance(mock_top_k_tokens[0][0], tuple)
    assert isinstance(mock_top_k_tokens[0][0][0], str)
    assert isinstance(mock_top_k_tokens[0][0][1], float)
    assert len(mock_top_k_tokens) == 2
    assert len(mock_top_k_tokens[0]) == 3
    assert len(mock_top_k_tokens[1]) == 3
