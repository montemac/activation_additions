"""Unit tests for the `sparse_coding` submodule."""


from collections import defaultdict

import pytest
import torch as t
import transformers
from accelerate import Accelerator

from sparse_coding.utils.top_k import (
    calculate_effects,
    project_activations,
    # unpad_activations,
    # select_top_k_tokens,
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
    """Return mock token ids and autoencoder activations."""

    question_token_ids: list[list[int]] = [
        [0, 1, 2, 3, 4, 0],
        [9, 10, 11, 1, 2, 3],
    ]
    feature_activations: list[t.Tensor] = [t.randn(1024, 6) for _ in range(2)]

    return question_token_ids, feature_activations


def test_calculate_effects(  # pylint: disable=redefined-outer-name
    mock_autoencoder, mock_data
):
    """Test the `calculate_effects` function."""

    # Pytest fixture injections.
    mock_encoder, tokenizer, accelerator = mock_autoencoder
    question_token_ids, feature_activations = mock_data

    batch_size = 2

    mock_effects = calculate_effects(
        question_token_ids,
        feature_activations,
        mock_encoder,
        tokenizer,
        accelerator,
        batch_size,
    )

    assert isinstance(mock_effects, defaultdict)
    assert isinstance(mock_effects[0], defaultdict)
    assert len(mock_effects) == 2  # Two question activations.
    assert len(mock_effects[0]) == 8  # Eight unique tokens.


def test_project_activations(  # pylint: disable=redefined-outer-name
    mock_autoencoder,
):
    """Test the `project_activations` function."""

    acts_list = [t.randn(5, 512) for _ in range(2)]
    mock_encoder, _, accelerator = mock_autoencoder

    mock_projections = project_activations(
        acts_list, mock_encoder, accelerator
    )

    assert isinstance(mock_projections, list)
    assert isinstance(mock_projections[0], t.Tensor)
    assert mock_projections[0].shape == (5, 1024)


# def test_unpad_activations():
#     pass


# def test_select_top_k_tokens():
#     pass
