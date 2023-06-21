# %%
""" Test suite for logging.py """

import pytest

import pandas as pd
from transformer_lens import HookedTransformer

from activation_additions import (
    logging,
    completion_utils,
    prompt_utils,
    utils,
)

utils.enable_ipython_reload()


@pytest.fixture(name="model")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer used
    for fast logging testing."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-2l", device="cpu"
    )


# In order for these tests to work, you must have a wandb account and
# have set up your wandb API key.  See https://docs.wandb.ai/quickstart
def test_logging(model):
    """Tests a sweep over prompts with logging enabled.  Verifies that
    the correct data is uploaded to a new wandb run."""
    # TODO: do this properly with pytest config
    pytest.skip("Logging testing is slow! Change this line to enable it.")
    # Perform a completion test
    results: pd.DataFrame = completion_utils.gen_using_activation_additions(
        model=model,
        activation_additions=[
            prompt_utils.ActivationAddition(
                prompt="Love",
                act_name=prompt_utils.get_block_name(block_num=0),
                coeff=1.0,
            ),
            prompt_utils.ActivationAddition(
                prompt="Fear",
                act_name=prompt_utils.get_block_name(block_num=0),
                coeff=-1.0,
            ),
        ],
        prompt_batch=["This is a test", "Let's talk about"],
        log={"tags": ["test"], "notes": "testing"},
    )
    # Download the artifact data and convert to a DataFrame
    results_logged = logging.get_objects_from_run(
        logging.last_run_info["path"],
    )["gen_using_activation_additions"]

    print(results, results_logged)

    # Compare with the reference DataFrame
    pd.testing.assert_frame_equal(results, results_logged)


def test_positional_args(model):
    """Function test a call to a loggable function using positional
    arguments, which were initially not supported by the loggable
    decorator."""
    completion_utils.print_n_comparisons(
        "I think you're ",
        model,
        num_comparisons=5,
        activation_additions=[],
        seed=0,
    )
