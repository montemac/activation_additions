"""Basic demonstration of sweeps and metrics operation."""

# %%
# Imports, etc.
import pickle

import numpy as np
import torch

import plotly.express as px

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    sweeps,
    metrics,
    prompt_utils,
    completion_utils,
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
# Generate some example completions, for reproduction reference from
# Alex's notebook.
weddings_prompts = [
    *prompt_utils.get_x_vector(
        prompt1="I always talk about weddings",
        prompt2="I never talk about weddings",
        coeff=4,
        act_name=6,
        pad_method="tokens_right",
        model=MODEL,
        custom_pad_id=MODEL.to_single_token(" "),
    )
]

completion_utils.print_n_comparisons(
    model=MODEL,
    prompt="Frozen starts off with a scene about",
    tokens_to_generate=50,
    rich_prompts=weddings_prompts,
    num_comparisons=7,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)


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
    "Frozen starts off with a scene about",
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
# Run the sweep of completions, or load from cache
CACHE_FN = "sweeps_demo_cache.pkl"
try:
    with open(CACHE_FN, "rb") as file:
        normal_df, patched_df, rich_prompts_df = pickle.load(file)
except FileNotFoundError:
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
    with open(CACHE_FN, "wb") as file:
        pickle.dump((normal_df, patched_df, rich_prompts_df), file)

# %%
# Visualize
# Reduce the patched DataFrame
reduced_df = patched_df.groupby(["prompts", "rich_prompt_index"]).mean(
    numeric_only=True
)
reduced_joined_df = reduced_df.join(
    rich_prompts_df, on="rich_prompt_index"
).reset_index()

# Reduce the normal DataFrame
reduced_normal_df = normal_df.groupby(["prompts"]).mean(numeric_only=True)


# Plot function
def plot_col(
    data,
    col_to_plot,
    title,
    col_x="coeff",
    col_color="act_name",
    baseline_data=None,
):
    """Plot a column, with colors/facets/x set."""
    fig = px.line(
        data,
        title=title,
        color=col_color,
        y=col_to_plot,
        x=col_x,
        facet_col="prompts",
    )
    if baseline_data is not None and col_to_plot in baseline_data:
        for ii, prompt in enumerate(baseline_data.index):
            fig.add_hline(
                y=baseline_data.loc[prompt][col_to_plot],
                row=1,
                col=ii + 1,
                annotation_text="normal",
                annotation_position="bottom left",
            )
    fig.show()


# plot_col(
#     reduced_joined_df, "wedding_words_count", "Average wedding word count"
# )
# plot_col(reduced_joined_df, "loss", "Average loss")

# Exlude the extreme coeffs, likely not that interesting
reduced_joined_filt_df = reduced_joined_df[
    (reduced_joined_df["coeff"] >= -4) & (reduced_joined_df["coeff"] <= 4)
]

# Plot
plot_col(
    reduced_joined_filt_df,
    "wedding_words_count",
    "Average wedding word count",
    baseline_data=reduced_normal_df,
)
plot_col(
    reduced_joined_filt_df,
    "loss",
    "Average loss",
    baseline_data=reduced_normal_df,
)
plot_col(
    reduced_joined_filt_df,
    "wedding_words_count",
    "Average wedding word count",
    col_x="act_name",
    col_color="coeff",
    baseline_data=reduced_normal_df,
)
plot_col(
    reduced_joined_filt_df,
    "loss",
    "Average loss",
    col_x="act_name",
    col_color="coeff",
    baseline_data=reduced_normal_df,
)
