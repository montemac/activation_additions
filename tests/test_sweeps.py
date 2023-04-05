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
from algebraic_value_editing import prompt_utils
from algebraic_value_editing.prompt_utils import RichPrompt


@pytest.fixture
def model() -> HookedTransformer:
    return HookedTransformer.from_pretrained(model_name="gpt2-small")


def test_make_rich_prompts():
    rich_prompts_df = sweeps.make_rich_prompts(
        [[("Good", 1.0), ("Bad", -1.0)], [("Amazing", 2.0)]],
        [prompt_utils.get_block_name(block_num=num) for num in [6, 7, 8]],
        np.array([1.0, 5, 10.0, 20.0]),
    )
    target = pd.DataFrame(
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
    pd.testing.assert_frame_equal(rich_prompts_df, target)


SWEEP_OVER_PROMPTS_CACHE_FN = "sweep_over_prompts_cache.pkl"


def test_sweep_over_prompts(model):
    act_name = prompt_utils.get_block_name(block_num=6)
    normal_df, patched_df = sweeps.sweep_over_prompts(
        model,
        [
            "Roses are red, violets are blue",
            "The most powerful emotion is",
            "I feel",
        ],
        [
            [
                RichPrompt(1.0, act_name, "Love"),
                RichPrompt(-1.0, act_name, "Fear"),
            ],
            [
                RichPrompt(10.0, act_name, "Love"),
                RichPrompt(-10.0, act_name, "Fear"),
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
