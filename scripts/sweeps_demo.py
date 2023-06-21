"""Basic demonstration of sweeps and metrics operation."""

# %%
# Imports, etc.
import pickle

import numpy as np
import torch

from transformer_lens import HookedTransformer

from activation_additions import (
    sweeps,
    metrics,
    prompt_utils,
    completion_utils,
    utils,
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)

# %%
# Load a model
MODEL = HookedTransformer.from_pretrained(model_name="gpt2-xl", device="cpu")
_ = MODEL.to("cuda:0")

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
        custom_pad_id=int(MODEL.to_single_token(" ")),
    )
]

completion_utils.print_n_comparisons(
    model=MODEL,
    prompt="Frozen starts off with a scene about",
    tokens_to_generate=50,
    activation_additions=weddings_prompts,
    num_comparisons=7,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)


# %%
# Generate a set of ActivationAdditions over a range of phrases, layers and
# coeffs
# TODO: need to find a way to add padding specifications to these sweep inputs
activation_additions_df = sweeps.make_activation_additions(
    [
        [
            ("Anger", 1.0),
            ("Calm", -1.0),
        ]
    ],
    [
        prompt_utils.get_block_name(block_num=num)
        for num in range(0, len(MODEL.blocks), 4)
    ],
    np.array([-4, -1, 1, 4]),
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
        normal_df, patched_df, activation_additions_df = pickle.load(file)
except FileNotFoundError:
    normal_df, patched_df = sweeps.sweep_over_prompts(
        MODEL,
        prompts,
        activation_additions_df["activation_additions"],
        num_normal_completions=100,
        num_patched_completions=100,
        seed=0,
        metrics_dict=metrics_dict,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
    )
    with open(CACHE_FN, "wb") as file:
        pickle.dump((normal_df, patched_df, activation_additions_df), file)

# %%
# Visualize

# Reduce data
reduced_normal_df, reduced_patched_df = sweeps.reduce_sweep_results(
    normal_df, patched_df, activation_additions_df
)

# Exlude the extreme coeffs, likely not that interesting
reduced_patched_filt_df = reduced_patched_df[
    (reduced_patched_df["coeff"] >= -4) & (reduced_patched_df["coeff"] <= 4)
]

# Plot

sweeps.plot_sweep_results(
    reduced_patched_filt_df,
    "wedding_words_count",
    "Average wedding word count",
    col_x="act_name",
    col_color="coeff",
    baseline_data=reduced_normal_df,
).show()
sweeps.plot_sweep_results(
    reduced_patched_filt_df,
    "loss",
    "Average loss",
    col_x="act_name",
    col_color="coeff",
    baseline_data=reduced_normal_df,
).show()

# %%
