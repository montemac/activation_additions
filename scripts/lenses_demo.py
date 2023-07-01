# %%

%load_ext autoreload
%autoreload 2

# %%
from typing import List, Dict, Callable, Literal
from transformer_lens.HookedTransformer import HookedTransformer

from tuned_lens import TunedLens
from tuned_lens.plotting import PredictionTrajectory, TrajectoryStatistic
import numpy as np
import torch

from algebraic_value_editing import completion_utils, utils
from algebraic_value_editing.prompt_utils import (
    ActivationAddition,
    get_x_vector,
)
from algebraic_value_editing.lenses import (
    run_hooked_and_normal_with_cache,
    prediction_trajectories,
)
import algebraic_value_editing.hook_utils as hook_utils
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM

import torch
import pandas as pd

utils.enable_ipython_reload()


# %%

model_name = "gpt2-xl"

if torch.has_cuda:
    device = torch.device("cuda", 1)
elif torch.has_mps:
    device = torch.device("cpu")  # mps not working yet
else:
    device = torch.device("cpu")

torch.set_grad_enabled(False)

# Load model from huggingface
# TODO: Fix memory waste from loading model twice
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # revision=f"checkpoint-{cfg.checkpoint_value}"
)

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name=model_name,
    hf_model=hf_model,
    device="cpu",
).to(device)
model.cfg.device = device
model.eval()

# %%

# NOTE: Hash mismatch on latest tuned lens. Seems fine to ignore, see issue:
# https://github.com/AlignmentResearch/tuned-lens/issues/89
tuned_lens = TunedLens.from_model_and_pretrained(
    hf_model, lens_resource_id=model_name
).to(device)

# %%
# Library helpers

Metric = Literal["entropy", "forward_kl", "max_probability"]


def apply_metric(metric: Metric, pt: PredictionTrajectory) -> TrajectoryStatistic:
    return getattr(pt, metric)()


def plot_lens_diff(
    caches: List[Dict[str, torch.Tensor]],
    dataframes: List[pd.DataFrame],
    metric: Metric,
    layer_stride: int = 4,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.03,
        # subplot_titles=("Entropy", "Forward KL", "Cross Entropy", "Max Probability"),
    )

    fig.update_layout(
        height=1000,
        width=800,
        title_text="Tokens visualized with the Tuned Lens",
    )

    trajectories = prediction_trajectories(
        caches, dataframes, model.tokenizer, tuned_lens
    )

    stats_normal = apply_metric(metric, trajectories[0])
    stats_modified = apply_metric(metric, trajectories[1])

    # Update heatmap data inside playground function
    hm_normal = stats_normal.heatmap(layer_stride=layer_stride)
    hm_modified = stats_modified.heatmap(layer_stride=layer_stride)

    # Fix colorbars to use the same scale (ty gpt4)
    zmax = max(stats_normal.stats.max(), stats_modified.stats.max())
    zmin = min(stats_normal.stats.min(), stats_modified.stats.min())
    hm_normal.update(zmin=zmin, zmax=zmax, showscale=False)
    hm_modified.update(zmin=zmin, zmax=zmax, showscale=True)

    fig.add_trace(hm_normal, row=1, col=1)
    fig.add_trace(hm_modified, row=2, col=1)
    return fig


# Main playground for lenses. Run with ctrl+enter

# Steering vector: "Love" - "Hate" before attention layer 6 with coefficient +5
setup = dict(
    prompt = "I hate you because you're a wonderful person",
    xv=dict(
        prompt1 = "Love",
        prompt2 = "Hate",
        coeff = 5,
        act_name = 6,
    )
)

# Steering vector: "Bush did 9/11 because" - "      " before attention layer 23 with coefficient +1
# setup = dict(
#     prompt = "Barack Obama was born in a secret CIA prison",
#     xv=dict(
#         prompt1="Bush did 9/11 because",
#         prompt2="",
#         coeff = 1,
#         act_name = 23,
#     )
# )

# Steering vector: "Intent to praise" - "Intent to hurt" before attention layer 6 with coefficient +15
# setup = dict(
#     prompt="I want to kill you because you're such a",
#     xv=dict(
#         prompt1="Intent to praise",
#         prompt2="Intent to hurt",
#         coeff = 15,
#         act_name = 6,
#     )
# )


activation_additions = [
    *get_x_vector(
        **setup["xv"],
        pad_method="tokens_right",
        model=model,
        custom_pad_id=model.to_single_token(" "),
    )
]

dataframes, caches = run_hooked_and_normal_with_cache(
    model=model,
    activation_additions=activation_additions,
    gen_args=dict(
        prompt_batch=[setup['prompt']] * 1, tokens_to_generate=0, top_p=0.3, seed=0 # NOTE: topp, seed don't matter if not generating.
    ),
)

trajectories = prediction_trajectories(
    caches, dataframes, model.tokenizer, tuned_lens
)

fig = plot_lens_diff(
    caches=caches,
    dataframes=dataframes,
    metric="cross_entropy",
    layer_stride=2,
)
fig.show()

# %%
# Play with printing completions to check behavior


completion_utils.print_n_comparisons(
    prompt=prompt,
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    # freq_penalty=1,
    top_p=0.8,
    tokens_to_generate=8,
)

# %%
