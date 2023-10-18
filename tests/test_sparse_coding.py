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
    """Return mock input token ids by q and encoder activations by q."""

    # "Just say, oops."
    # "Just say, hello world!"
    input_token_ids_by_q: list[list[int]] = [
        [6300, 1333, 13, 258, 2695, 15],
        [6300, 1333, 13, 23120, 1533, 2],
    ]
    encoder_activations_by_q_block: list[t.Tensor] = [
        (t.ones(6, 1024)) * 7,
        (t.ones(6, 1024)) * 11,
    ]

    return input_token_ids_by_q, encoder_activations_by_q_block


def test_per_input_token_effects(  # pylint: disable=redefined-outer-name
    mock_autoencoder, mock_data
):
    """Test `per_input_token_effects`."""

    # Pytest fixture injections.
    mock_encoder, tokenizer, accelerator = mock_autoencoder
    question_token_ids, feature_activations = mock_data

    dims_in_batch = 200
    large_model_mode = False

    mock_effects = per_input_token_effects(
        question_token_ids,
        feature_activations,
        mock_encoder,
        tokenizer,
        accelerator,
        dims_in_batch,
        large_model_mode,
    )

    try:
        # Structural asserts.
        assert isinstance(mock_effects, defaultdict)
        assert isinstance(mock_effects[0], defaultdict)
        assert len(mock_effects) == 1024  # 1024 encoder dimensions.
        assert len(mock_effects[0]) == 9  # 9 unique tokens.
        # Semantic asserts.
        assert mock_effects[0]["Just"] == (7 + 11) / 2
        assert mock_effects[100]["Ġsay"] == (7 + 11) / 2
        assert mock_effects[200][","] == (7 + 11) / 2

        assert mock_effects[0]["Ġo"] == 7
        assert mock_effects[100]["ops"] == 7
        assert mock_effects[200]["."] == 7

        assert mock_effects[0]["Ġhello"] == 11
        assert mock_effects[100]["Ġworld"] == 11
        assert mock_effects[200]["!"] == 11

    except Exception as e:  # pylint: disable=broad-except
        pytest.fail(
            f"`per_input_token_effects` failed unit test with error: {e}"
        )


def test_project_activations(  # pylint: disable=redefined-outer-name
    mock_autoencoder,
):
    """Test `project_activations`."""

    acts_list = [t.randn(5, 512) for _ in range(2)]
    mock_encoder, _, accelerator = mock_autoencoder

    mock_projections = project_activations(
        acts_list, mock_encoder, accelerator
    )

    try:
        assert isinstance(mock_projections, list)
        assert isinstance(mock_projections[0], t.Tensor)
        assert mock_projections[0].shape == (5, 1024)
    except Exception as e:  # pylint: disable=broad-except
        pytest.fail(f"`project_activations` failed unit test with error: {e}")


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
    try:
        assert isinstance(mock_top_k_tokens, defaultdict)
        assert isinstance(mock_top_k_tokens[0], list)
        assert isinstance(mock_top_k_tokens[0][0], tuple)
        assert isinstance(mock_top_k_tokens[0][0][0], str)
        assert isinstance(mock_top_k_tokens[0][0][1], float)
        assert len(mock_top_k_tokens) == 2
        assert len(mock_top_k_tokens[0]) == 3
        assert len(mock_top_k_tokens[1]) == 3
    except Exception as e:  # pylint: disable=broad-except
        pytest.fail(f"`select_top_k_tokens` failed unit test with error: {e}")
