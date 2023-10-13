"""Unit tests for the `sparse_coding` submodule."""


from collections import defaultdict

import pytest
import torch as t
import transformers
from accelerate import Accelerator

from sparse_coding.utils.top_k import (
    calculate_effects,
    project_activations,
    unpad_activations,
    select_top_k_tokens,
)


# Test determinism.
t.manual_seed(0)


@pytest.fixture
def mock_model():
    """Return a mock model, its tokenizer, and its accelerator."""

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m"
    )
    accelerator = Accelerator()

    return model, tokenizer, accelerator


@pytest.fixture
def mock_data():
    """Return mock token ids and autoencoder activations."""

    question_token_ids: list[list[int]] = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
    feature_activations: list[t.Tensor] = [t.randn(1024, 5) * 2]

    return question_token_ids, feature_activations


def test_calculate_effects(  # pylint: disable=redefined-outer-name
    mock_model, mock_data
):
    """Test the `calculate_effects` function."""

    # Pytest fixture injections.
    model, tokenizer, accelerator = mock_model
    question_token_ids, feature_activations = mock_data

    batch_size = 2

    mock_effects = calculate_effects(
        question_token_ids,
        feature_activations,
        model,
        tokenizer,
        accelerator,
        batch_size,
    )

    assert isinstance(mock_effects, defaultdict)
    assert isinstance(mock_effects[0], defaultdict)
    assert len(mock_effects) == 1024


def test_project_activations(  # pylint: disable=redefined-outer-name
    mock_model,
):
    """Test the `project_activations` function."""

    acts_list = [t.randn(5, 512) * 2]
    mock_autoencoder = t.nn.Linear(512, 1024)
    _, __, accelerator = mock_model

    mock_projections = project_activations(
        acts_list, mock_autoencoder, accelerator
    )

    assert isinstance(mock_projections, list)
    assert isinstance(mock_projections[0], t.Tensor)
    assert mock_projections[0].shape == (1024, 5)


# def test_unpad_activations():
#     pass


# def test_select_top_k_tokens():
#     pass
