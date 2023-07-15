# %%
"""Test suite for logits.py"""
import pytest

import torch

from transformer_lens import HookedTransformer

from activation_additions import utils, experiments

utils.enable_ipython_reload()


@pytest.fixture(name="model")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer used
    for fast metric testing."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-2l", device="cpu"
    )


def test_get_token_probs(model):
    """Test get_token_probs() function."""
    avg_logprob, perplexity, logprobs = experiments.get_stats_over_corpus(
        model=model, corpus_texts=["This is a test sentence."]
    )
    assert avg_logprob == pytest.approx(-5.2312, abs=1e-4)
    assert perplexity == pytest.approx(187.008480, abs=1e-4)
    assert torch.allclose(
        logprobs,
        torch.tensor([-7.7388, -1.4853, -1.3223, -6.7956, -12.2510, -1.7938]),
        atol=1e-4,
    )
    avg_logprob_mask_len, _, _ = experiments.get_stats_over_corpus(
        model=model,
        corpus_texts=["This is a test sentence."],
        mask_len=2,
    )
    assert avg_logprob_mask_len == pytest.approx(logprobs[2:].mean(), abs=1e-4)
