""" Test suite for sweeps.py """
# %%
import pickle
from typing import Tuple
import os

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from transformer_lens import HookedTransformer

from activation_additions import sweeps, prompt_utils, utils
from activation_additions.prompt_utils import ActivationAddition

utils.enable_ipython_reload()

# Filename for pre-pickled assets
SWEEP_OVER_PROMPTS_CACHE_FN: str = "tests/sweep_over_prompts_cache.pkl"

# GPU sometimes produces different completions than CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture(name="model")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer used
    for fast sweep testing."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-2l", device="cpu"
    )


def load_cached_sweep_over_prompts():
    """Load pre-saved sweep results."""
    with open(SWEEP_OVER_PROMPTS_CACHE_FN, "rb") as file:
        return pickle.load(file)


def test_make_activation_additions():
    """Test for make_activation_additions() function.  Provides a simple set of
    phrases+coeffs, activation names and additional coeffs that the
    function under test will expand into all permutations, in the style
    of np.ndgrid.  The return value is compared against a pre-prepared
    reference output."""
    # Call the function under test
    activation_additions_df: pd.DataFrame = sweeps.make_activation_additions(
        phrases=[[("Good", 1.0), ("Bad", -1.0)], [("Amazing", 2.0)]],
        act_names=[
            prompt_utils.get_block_name(block_num=num) for num in [6, 7, 8]
        ],
        coeffs=np.array([1.0, 5, 10.0, 20.0]),
    )
    # Compre to pre-defined target
    pd.testing.assert_frame_equal(
        activation_additions_df, MAKE_ACTIVATION_ADDITIONS_TARGET
    )


def do_sweep(
    model: HookedTransformer, **sweep_kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to perform a small example sweep and return
    the resulting DataFrames"""
    act_name: str = prompt_utils.get_block_name(block_num=0)
    activation_additions_df = sweeps.make_activation_additions(
        phrases=[[("Love", 1.0), ("Fear", -1.0)]],
        act_names=[act_name],
        coeffs=np.array([1.0, 10.0]),
    )
    normal_df, patched_df = sweeps.sweep_over_prompts(
        model=model,
        prompts=[
            "Roses are red, violets are blue",
            "The most powerful emotion is",
            "I feel",
        ],
        activation_additions=activation_additions_df["activation_additions"],
        num_normal_completions=4,
        num_patched_completions=4,
        temperature=0.0,
        **sweep_kwargs,
    )
    return normal_df, patched_df, activation_additions_df


def make_and_save_sweep_results(model):
    """Convenience function for performing a sweep and cache the
    results, which is required any time a breaking change is made that
    would render the existing cached targets incorrect.  New target
    results saved by this function should be verified for correctness
    when they are created."""
    normal_df, patched_df, activation_additions_df = do_sweep(model)
    reduced_normal_df, reduced_patched_df = sweeps.reduce_sweep_results(
        normal_df, patched_df, activation_additions_df
    )
    with open(SWEEP_OVER_PROMPTS_CACHE_FN, "wb") as file:
        pickle.dump(
            (
                normal_df,
                patched_df,
                activation_additions_df,
                reduced_normal_df,
                reduced_patched_df,
            ),
            file,
        )


def test_sweep_over_prompts(model):
    """Test for sweep_over_prompts().  Uses a toy model fixture, passes
    a handful of ActivationAdditions and prompts, and compares results to a
    pre-cached reference output."""
    normal_df, patched_df, _ = do_sweep(model)
    normal_target, patched_target, _, _, _ = load_cached_sweep_over_prompts()
    pd.testing.assert_frame_equal(normal_df, normal_target)
    pd.testing.assert_frame_equal(patched_df, patched_target)


def test_reduce_sweep_results():
    """Test for reduce_sweep_results().  Uses a pre-prepared set of
    sweep results as input, and pre-calculated reduction results as test
    target."""
    # Load sweep results and target reduction results
    (
        normal_df,
        patched_df,
        activation_additions_df,
        reduced_normal_target,
        reduced_patched_target,
    ) = load_cached_sweep_over_prompts()

    # Reduce DataFrames and compare reductions to target
    reduced_normal_df, reduced_patched_df = sweeps.reduce_sweep_results(
        normal_df, patched_df, activation_additions_df
    )
    pd.testing.assert_frame_equal(reduced_normal_df, reduced_normal_target)
    pd.testing.assert_frame_equal(reduced_patched_df, reduced_patched_target)


def test_plot_sweep_results():
    """Test for plot_sweep_results(). Doesn't verify visual correctness
    of plot, just verifies that the function funs without exceptions
    with various calling signatures and returns the correct object
    type."""
    (
        _,
        _,
        _,
        reduced_normal_df,
        reduced_patched_df,
    ) = load_cached_sweep_over_prompts()

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
MAKE_ACTIVATION_ADDITIONS_TARGET = pd.DataFrame(
    {
        "activation_additions": [
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=1.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-1.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=5.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-5.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=10.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-10.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=20.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-20.0,
                    act_name="blocks.6.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=1.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-1.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=5.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-5.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=10.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-10.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=20.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-20.0,
                    act_name="blocks.7.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=1.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-1.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=5.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-5.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=10.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-10.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Good",
                    coeff=20.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
                ActivationAddition(
                    prompt="Bad",
                    coeff=-20.0,
                    act_name="blocks.8.hook_resid_pre",
                ),
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=2.0,
                    act_name="blocks.6.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=10.0,
                    act_name="blocks.6.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=20.0,
                    act_name="blocks.6.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=40.0,
                    act_name="blocks.6.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=2.0,
                    act_name="blocks.7.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=10.0,
                    act_name="blocks.7.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=20.0,
                    act_name="blocks.7.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=40.0,
                    act_name="blocks.7.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=2.0,
                    act_name="blocks.8.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=10.0,
                    act_name="blocks.8.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
                    prompt="Amazing",
                    coeff=20.0,
                    act_name="blocks.8.hook_resid_pre",
                )
            ],
            [
                ActivationAddition(
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
