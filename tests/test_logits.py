# %%
"""Test suite for logits.py"""
import pytest

import pandas as pd
import numpy as np

from transformer_lens import HookedTransformer

from activation_additions import utils, logits

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
    probs = logits.get_token_probs(model, "My name is")
    assert isinstance(probs, pd.DataFrame)
    assert probs.shape == (4, 96524)
    assert probs.columns.levels[0].to_list() == ["probs", "logprobs"]  # type: ignore
    alice_token = int(model.to_single_token(" Alice"))
    bob_token = int(model.to_single_token(" Bob"))
    k_token = int(model.to_single_token(" K"))
    assert np.allclose(probs.iloc[-1].loc[("probs", alice_token)], 0.000495836)
    assert np.allclose(probs.iloc[-1].loc[("probs", bob_token)], 0.002270653)
    assert np.allclose(probs.iloc[-1].loc[("probs", k_token)], 0.01655953)


# TODO: write more tests!
