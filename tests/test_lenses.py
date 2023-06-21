# %%
""" Test suite for lenses.py """

import pytest
import torch

from transformer_lens import HookedTransformer
from tuned_lens import TunedLens
from transformers import AutoModelForCausalLM

from activation_additions.prompt_utils import get_x_vector
from activation_additions import lenses, utils

utils.enable_ipython_reload()

# smallest tuned lens supported model
MODEL = "EleutherAI/pythia-70m-deduped"


@pytest.fixture(name="model")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer used
    for fast logging testing."""

    torch.set_grad_enabled(False)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL)
    model = HookedTransformer.from_pretrained(
        model_name=MODEL, hf_model=hf_model, device="cpu"
    )
    model.hf_model = hf_model
    model.eval()
    return model


@pytest.fixture(name="tuned_lens")
def fixture_tuned_lens(model: HookedTransformer) -> lenses.TunedLens:
    """Test fixture that returns a small pre-trained transformer used
    for fast logging testing."""
    return TunedLens.from_model_and_pretrained(
        model.hf_model, lens_resource_id=MODEL, map_location="cpu"  # type: ignore
    ).to("cpu")


def test_lenses(model, tuned_lens):
    """
    Checks no exceptions are raised when using lenses are intended.
    """

    prompt = "I hate you because"

    activation_additions = [
        *get_x_vector(
            prompt1="Love",
            prompt2="Hate",
            coeff=5,
            act_name=2,
            pad_method="tokens_right",
            model=model,
            custom_pad_id=model.to_single_token(" "),
        )
    ]

    dataframes, caches = lenses.run_hooked_and_normal_with_cache(
        model=model,
        activation_additions=activation_additions,
        gen_args={"prompt_batch": [prompt] * 1, "seed": 0},
    )

    _ = lenses.prediction_trajectories(
        caches, dataframes, model.tokenizer, tuned_lens
    )
