# %%
try:
    get_ipython().__class__.__name__
    is_ipython = True
except:
    is_ipython = False
if is_ipython:
    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import pytest
import numpy as np
import pandas as pd
import pandas.testing
import pickle

from transformer_lens import HookedTransformer

from algebraic_value_editing import sweeps
from algebraic_value_editing import hook_utils
from algebraic_value_editing.rich_prompts import RichPrompt


@pytest.fixture
def model() -> HookedTransformer:
    return HookedTransformer.from_pretrained(model_name="gpt2-small")


SWEEP_OVER_PROMPTS_CACHE_FN = "sweep_over_prompts_cache.pkl"


def test_sweep_over_prompts(model):
    act_name = hook_utils.get_block_name(block_num=6)
    normal_df, patched_df = sweeps.sweep_over_prompts(
        model,
        [
            "Roses are red, violets are blue",
            "The most powerful emotion is",
            "I feel",
        ],
        [
            [
                RichPrompt("Love", 1.0, act_name),
                RichPrompt("Fear", -1.0, act_name),
            ],
            [
                RichPrompt("Love", 10.0, act_name),
                RichPrompt("Fear", -10.0, act_name),
            ],
        ],
        num_normal_completions=4,
        num_patched_completions=4,
        seed=42,
    )
    with open(SWEEP_OVER_PROMPTS_CACHE_FN, "rb") as fl:
        normal_target, patched_target = pickle.load(fl)
    pd.testing.assert_frame_equal(normal_df, normal_target)
    pd.testing.assert_frame_equal(patched_df, patched_target)
