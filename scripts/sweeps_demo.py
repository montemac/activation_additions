"""Basic demonstration of sweeps and metrics operation."""

# %%
# Imports, etc.
import numpy as np
import torch

import plotly.express as px

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    sweeps,
    metrics,
    prompt_utils,
)

try:
    from IPython import get_ipython

    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except AttributeError:
    pass

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)

# %%
# Load a model
MODEL = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to("cuda:0")


# %%
# Generate a set of RichPrompts over a range of phrases, layers and
# coeffs
# TODO: need to find a way to add padding specifications to these sweep inputs
rich_prompts_df = sweeps.make_rich_prompts(
    [
        [
            ("I talk about weddings constantly  ", 1.0),
            ("I do not talk about weddings constantly", -1.0),
        ]
    ],
    [
        prompt_utils.get_block_name(block_num=num)
        for num in range(0, len(MODEL.blocks), 4)
    ],
    np.array([-64, -16, -4, -1, 1, 4, 16, 64]),
)

# %%
# Populate a list of prompts to complete
prompts = [
    "I went up to my friend and said",
    "Batman Begins starts off with a scene about",
]

# %%
# Create metrics
metrics_dict = {
    "wedding_words": metrics.get_word_count_metric(
        [
            "wedding",
            "weddings",
            "wed",
            "marry",
            "married",
            "marriage",
            "bride",
            "groom",
            "honeymoon",
        ]
    ),
}


# %%
# Run the sweep of completions
normal_df, patched_df = sweeps.sweep_over_prompts(
    MODEL,
    prompts,
    rich_prompts_df["rich_prompts"],
    num_normal_completions=100,
    num_patched_completions=100,
    seed=0,
    metrics_dict=metrics_dict,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)

# %%
# Visualize
reduced_df = patched_df.groupby(["prompts", "rich_prompt_index"]).mean(
    numeric_only=True
)
reduced_joined_df = reduced_df.join(
    rich_prompts_df, on="rich_prompt_index"
).reset_index()


def plot_col(data, col_to_plot, title):
    """Plot a column, with colors/facets/x set."""
    px.line(
        data,
        title=title,
        x="coeff",
        y=col_to_plot,
        color="act_name",
        facet_col="prompts",
    ).show()


plot_col(
    reduced_joined_df, "wedding_words_count", "Average wedding word count"
)
plot_col(reduced_joined_df, "loss", "Average loss")

reduced_joined_filt_df = reduced_joined_df[
    (reduced_joined_df["coeff"] >= -4) & (reduced_joined_df["coeff"] <= 4)
]
plot_col(
    reduced_joined_filt_df, "wedding_words_count", "Average wedding word count"
)
plot_col(reduced_joined_filt_df, "loss", "Average loss")

# %%
# For reproduction reference from Alex's notebook
# default_kwargs = {"temperature": 1, "freq_penalty": 1, "top_p": 0.3}

# weddings_prompts_4 = [
#     *prompt_utils.get_x_vector(
#         prompt1="I talk about weddings constantly",
#         prompt2="I do not talk about weddings constantly",
#         coeff=4,
#         act_name=20,
#         pad_method="tokens_right",
#         model=MODEL,
#         custom_pad_id=MODEL.to_single_token(" "),
#     )
# ]

# completion_utils.print_n_comparisons(
#     model=MODEL,
#     prompt="I went up to my friend and said",
#     tokens_to_generate=100,
#     rich_prompts=weddings_prompts_4,
#     num_comparisons=15,
#     **default_kwargs,
#     seed=0,
# )
