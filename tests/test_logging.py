# %%
import pytest

import wandb
from transformer_lens import HookedTransformer

from algebraic_value_editing import logging

import test_sweeps

try:
    from IPython import get_ipython

    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except AttributeError:
    pass


@pytest.fixture(name="model")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer used
    for fast logging testing."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-2l", device="cpu"
    )


def test_sweep_with_logging(model):
    """Tests a sweep over prompts with logging enabled.  Verifies that
    the correct data is uploaded to a new wandb run."""
    # Perform a sweep
    normal_df, patched_df, rich_prompts_df = test_sweeps.do_sweep(
        model, log={"tags": ["test"], "notes": "testing"}
    )
    # Download the artifact data
    api = wandb.Api()
    run = api.run(logging.last_run_info["path"])
    # download_path = "tests/"
