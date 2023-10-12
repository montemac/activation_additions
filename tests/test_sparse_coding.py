"""Unit tests for the `sparse_coding` submodule."""


from collections import defaultdict

import torch as t
import transformers
from accelerate import Accelerator

from sparse_coding.utils.top_k import calculate_effects, project_activations
from sparse_coding.acts_collect import pad_activations
from sparse_coding.autoencoder import padding_mask
from sparse_coding.feature_tokens import unpad_activations, select_top_k_tokens


def test_calulate_effects():
    """Test the `calculate_effects` function."""

    question_token_ids: list[list[int]] = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
    feature_activations: list[t.Tensor] = []
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m"
    )
    accelerator = Accelerator()
    batch_size = 2

    effects = calculate_effects(
        question_token_ids,
        feature_activations,
        model,
        tokenizer,
        accelerator,
        batch_size,
    )

    assert isinstance(effects, defaultdict)
    assert len(effects) ==


def test_project_activations():
    pass


def test_pad_activations():
    pass


def test_padding_mask():
    pass


def test_unpad_activations():
    pass


def test_select_top_k_tokens():
    pass
