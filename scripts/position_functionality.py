""" Compare different settings for where we add the steering vector, in
terms of the residual streams to which activations are added. """
# %%
from typing import List, Dict, Callable
import pandas as pd
import torch
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import completion_utils, utils
from activation_additions.prompt_utils import (
    ActivationAddition,
    get_x_vector,
)

utils.enable_ipython_reload()


# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda")
_ = torch.set_grad_enabled(False)

# %%
sampling_kwargs = {"temperature": 1, "top_p": 0.3, "freq_penalty": 1.0}

wedding_additions: List[ActivationAddition] = [
    ActivationAddition(prompt=" wedding", coeff=4.0, act_name=6),
    ActivationAddition(prompt=" ", coeff=-4.0, act_name=6),
]
# %% Print out qualitative results
for location in ("front", "mid", "back"):
    print(completion_utils.bold_text(f"\nLocation: {location}"))
    completion_utils.print_n_comparisons(
        prompt=("I went up to my friend and said"),
        num_comparisons=10,
        addition_location=location,
        model=model,
        activation_additions=wedding_additions,
        seed=0,
        **sampling_kwargs,
    )

# %% Analyze how often wedding words show up under each condition

wedding_completions: int = 100

from activation_additions import metrics

metrics_dict: Dict[str, Callable] = {
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

dfs: List[pd.DataFrame] = []

for location in ("front", "mid", "back"):
    location_df: pd.DataFrame = (
        completion_utils.gen_using_activation_additions(
            model=model,
            prompt_batch=["I went up to my friend and said"]
            * wedding_completions,
            activation_additions=wedding_additions,
            addition_location=location,
            seed=0,
            **sampling_kwargs,
        )
    )

    # Store the fraction of dims we modified
    location_df["location"] = location
    dfs.append(location_df)

merged_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

# Store how many wedding words are present for each completion
merged_df = metrics.add_metric_cols(data=merged_df, metrics_dict=metrics_dict)

# %% [markdown]
# The "back" completions are less coherent, especially in the token
# immediately following the prompt. This is likely because the forward pass
# is getting modified just before that position. In our experience,
# directly modified positions have extremely different distributions
# over output token logits.
#
# Let's see how many wedding words are present, on average, for each
# addition location.

# %% Plot the average number of wedding words for each condition
avg_words_df: pd.DataFrame = (
    merged_df.groupby("location").mean(numeric_only=True).reset_index()
)
print(avg_words_df)

# %%
import plotly.express as px
import plotly.graph_objects as go

fig: go.Figure = px.bar(
    avg_words_df,
    x="location",
    y="wedding_words_count",
    title=(
        "(Average # of wedding words in completions) vs (Addition location)"
    ),
    labels={
        "location": ("Where we added the steering vector"),
        "wedding_words_count": "Avg. # of wedding words",
    },
)

# Set x ordering to "front", "mid", "back"
fig.update_xaxes(categoryorder="array", categoryarray=["front", "mid", "back"])

fig.show()

# %%
