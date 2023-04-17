""" Test suite for sweeps.py """

# %%
import pickle
from typing import Tuple

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from transformer_lens import HookedTransformer

from algebraic_value_editing import sweeps
from algebraic_value_editing import prompt_utils
from algebraic_value_editing.prompt_utils import RichPrompt

try:
    from IPython import get_ipython

    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except AttributeError:
    pass

# Filename for pre-pickled assets
SWEEP_OVER_PROMPTS_CACHE_FN = "tests/sweep_over_prompts_cache.pkl"


@pytest.fixture(name="model")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer used
    for fast sweep testing."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-2l", device="cpu"
    )


def test_make_rich_prompts():
    """Test for make_rich_prompts() function.  Provides a simple set of
    phrases+coeffs, activation names and additional coeffs that the
    function under test will expand into all permutations, in the style
    of np.ndgrid.  The return value is compared against a pre-prepared
    reference output."""
    # Call the function under test
    rich_prompts_df = sweeps.make_rich_prompts(
        [[("Good", 1.0), ("Bad", -1.0)], [("Amazing", 2.0)]],
        [prompt_utils.get_block_name(block_num=num) for num in [6, 7, 8]],
        np.array([1.0, 5, 10.0, 20.0]),
    )
    # Compre to pre-defined target
    pd.testing.assert_frame_equal(rich_prompts_df, MAKE_RICH_PROMPTS_TARGET)


def do_sweep(
    model: HookedTransformer,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to perform a small example sweep and return
    the resulting DataFrames"""
    act_name = prompt_utils.get_block_name(block_num=0)
    rich_prompts_df = sweeps.make_rich_prompts(
        [[("Love", 1.0), ("Fear", -1.0)]],
        [act_name],
        np.array([1.0, 10.0]),
    )
    normal_df, patched_df = sweeps.sweep_over_prompts(
        model=model,
        prompts=[
            "Roses are red, violets are blue",
            "The most powerful emotion is",
            "I feel",
        ],
        rich_prompts=rich_prompts_df["rich_prompts"],
        num_normal_completions=4,
        num_patched_completions=4,
        seed=42,
    )
    return normal_df, patched_df, rich_prompts_df


def make_and_save_sweep_results(model):
    """Convenience function for performing a sweep and cache the
    results, which is required any time a breaking change is made that
    would render the existing cached targets incorrect.  New target
    results saved by this function should be verified for correctness
    when they are created."""
    normal_df, patched_df, rich_prompts_df = do_sweep(model)
    reduced_normal_df, reduced_patched_df = sweeps.reduce_sweep_results(
        normal_df, patched_df, rich_prompts_df
    )
    with open(SWEEP_OVER_PROMPTS_CACHE_FN, "wb") as file:
        pickle.dump(
            (
                normal_df,
                patched_df,
                rich_prompts_df,
                reduced_normal_df,
                reduced_patched_df,
            ),
            file,
        )


def test_sweep_over_prompts(model):
    """Test for sweep_over_prompts().  Uses a toy model fixture, passes
    a handful of RichPrompts and prompts, and compares results to a
    pre-cached reference output."""
    normal_df, patched_df, _ = do_sweep(model)
    with open(SWEEP_OVER_PROMPTS_CACHE_FN, "rb") as file:
        normal_target, patched_target, _, _, _ = pickle.load(file)
    pd.testing.assert_frame_equal(normal_df, normal_target)
    pd.testing.assert_frame_equal(patched_df, patched_target)


def test_reduce_sweep_results():
    """Test for reduce_sweep_results().  Uses a pre-prepared set of
    sweep results as input, and pre-calculated reduction results as test
    target."""
    # Load sweep results and target reduction results
    with open(SWEEP_OVER_PROMPTS_CACHE_FN, "rb") as file:
        (
            normal_df,
            patched_df,
            rich_prompts_df,
            reduced_normal_target,
            reduced_patched_target,
        ) = pickle.load(file)
    # Reduce DataFrames and compare reductions to target
    reduced_normal_df, reduced_patched_df = sweeps.reduce_sweep_results(
        normal_df, patched_df, rich_prompts_df
    )
    pd.testing.assert_frame_equal(reduced_normal_df, reduced_normal_target)
    pd.testing.assert_frame_equal(reduced_patched_df, reduced_patched_target)


def test_plot_sweep_results():
    """Test for plot_sweep_results(). Doesn't verify visual correctness
    of plot, just verifies that the function funs without exceptions
    with various calling signatures and returns the correct object
    type."""
    with open(SWEEP_OVER_PROMPTS_CACHE_FN, "rb") as file:
        _, _, _, reduced_normal_df, reduced_patched_df = pickle.load(file)
    fig = sweeps.plot_sweep_results(
        data=reduced_patched_df,
        col_to_plot="loss",
        title="Testing",
        col_x="act_name",
        col_color="coeff",
        col_facet_col="prompts",
        baseline_data=reduced_normal_df,
    )
    assert isinstance(
        fig, go.Figure
    ), "plot_sweep_results() returned non-Figure() object"


# Assets
MAKE_RICH_PROMPTS_TARGET = pd.DataFrame(
    {
        "rich_prompts": [
            [
                RichPrompt(
                    prompt="Good",
                    coeff=1.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-1.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=5.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-5.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=10.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-10.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=20.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-20.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=1.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-1.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=5.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-5.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=10.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-10.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=20.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-20.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=1.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-1.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=5.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-5.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=10.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-10.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Good",
                    coeff=20.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
                RichPrompt(
                    prompt="Bad",
                    coeff=-20.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=2.0,
                    act_name="blocks.6.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=10.0,
                    act_name="blocks.6.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=20.0,
                    act_name="blocks.6.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=40.0,
                    act_name="blocks.6.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=2.0,
                    act_name="blocks.7.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=10.0,
                    act_name="blocks.7.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=20.0,
                    act_name="blocks.7.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=40.0,
                    act_name="blocks.7.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=2.0,
                    act_name="blocks.8.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=10.0,
                    act_name="blocks.8.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=20.0,
                    act_name="blocks.8.hook_resid_pre",
                )
            ],
            [
                RichPrompt(
                    prompt="Amazing",
                    coeff=40.0,
                    act_name="blocks.8.hook_resid_pre",
                )
            ],
        ],
        "phrases": [
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Good", 1.0), ("Bad", -1.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
            [("Amazing", 2.0)],
        ],
        "act_name": [
            "blocks.6.hook_resid_pre",
            "blocks.6.hook_resid_pre",
            "blocks.6.hook_resid_pre",
            "blocks.6.hook_resid_pre",
            "blocks.7.hook_resid_pre",
            "blocks.7.hook_resid_pre",
            "blocks.7.hook_resid_pre",
            "blocks.7.hook_resid_pre",
            "blocks.8.hook_resid_pre",
            "blocks.8.hook_resid_pre",
            "blocks.8.hook_resid_pre",
            "blocks.8.hook_resid_pre",
            "blocks.6.hook_resid_pre",
            "blocks.6.hook_resid_pre",
            "blocks.6.hook_resid_pre",
            "blocks.6.hook_resid_pre",
            "blocks.7.hook_resid_pre",
            "blocks.7.hook_resid_pre",
            "blocks.7.hook_resid_pre",
            "blocks.7.hook_resid_pre",
            "blocks.8.hook_resid_pre",
            "blocks.8.hook_resid_pre",
            "blocks.8.hook_resid_pre",
            "blocks.8.hook_resid_pre",
        ],
        "coeff": [
            1.0,
            5.0,
            10.0,
            20.0,
            1.0,
            5.0,
            10.0,
            20.0,
            1.0,
            5.0,
            10.0,
            20.0,
            1.0,
            5.0,
            10.0,
            20.0,
            1.0,
            5.0,
            10.0,
            20.0,
            1.0,
            5.0,
            10.0,
            20.0,
        ],
    }
)
