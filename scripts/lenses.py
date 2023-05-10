# %% 
%load_ext autoreload
%autoreload 2

# %%
from typing import List, Dict, Callable, Literal
from transformer_lens.HookedTransformer import HookedTransformer

from tuned_lens import TunedLens
from tuned_lens.plotting import PredictionTrajectory
from tqdm import tqdm
import numpy as np
import torch

from algebraic_value_editing import completion_utils
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector
from algebraic_value_editing.lenses import run_hooked_and_normal_with_cache, prediction_trajectories
import algebraic_value_editing.hook_utils as hook_utils
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM

import torch
import pandas as pd


# %%

model_name = 'gpt2-xl'

if torch.has_cuda:  device = torch.device('cuda', 1)
elif torch.has_mps: device = torch.device('cpu') # mps not working yet
else: device = torch.device('cpu')

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
tuned_lens = TunedLens.from_model_and_pretrained(hf_model, lens_resource_id=model_name).to(device)

# %%
# Library helpers

Metric = Literal['entropy', 'forward_kl', 'max_probability']


def apply_metric(metric: Metric, pt: PredictionTrajectory):
    return getattr(pt, metric)()


def plot_lens_diff(
    traj_normal: PredictionTrajectory,
    traj_modified: PredictionTrajectory,
    metric: Metric,
    layer_stride: int = 4,
    title_text: str = "Tokens visualized with the Tuned Lens",
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.03,
        # subplot_titles=("Entropy", "Forward KL", "Cross Entropy", "Max Probability"),
    )

    fig.update_layout(height=1000, width=800, title=title_text)

    # Update heatmap data inside playground function
    hm_normal = apply_metric(metric, traj_normal).heatmap(layer_stride=layer_stride)
    hm_modified = apply_metric(metric, traj_modified).heatmap(layer_stride=layer_stride)

    fig.add_trace(hm_normal, row=1, col=1)
    fig.add_trace(hm_modified, row=2, col=1)
    return fig


# Main playground for lenses. Run with ctrl+enter

coeffs = [2**i for i in range(20)]

traj_list_by_coeff: Dict[str, PredictionTrajectory] = {}

for coeff in tqdm(coeffs):
    prompt = "I hate you because"

    rich_prompts = [
        *get_x_vector(
            prompt1="Love",
            prompt2="Hate",
            coeff=coeff,
            act_name=6,
            pad_method="tokens_right",
            model=model,
            custom_pad_id=model.to_single_token(" "),
        )
    ]

    dataframes, caches = run_hooked_and_normal_with_cache(
        model=model, rich_prompts=rich_prompts,
        kw=dict(prompt_batch=[prompt] * 1, tokens_to_generate=0, top_p=0.3, seed=0),
    )

    traj_normal, traj_modified = prediction_trajectories(caches, dataframes, model.tokenizer, tuned_lens)
    traj_list_by_coeff[coeff] = traj_modified

    # fig = plot_lens_diff(
    #     traj_normal=traj_normal,
    #     traj_modified=traj_modified,
    #     metric='entropy',
    #     layer_stride=2,
    #     title_text=f"Tokens visualized with the Tuned Lens, coeff={coeff}",
    # )
    # fig.show()


# %%
# Animate traj modified heatmap for every coeff. use frames

import plotly.graph_objects as go

layer_stride = 4
hm_templ = traj_list_by_coeff[coeffs[0]].entropy().heatmap(layer_stride=layer_stride)
xmax, ymax = hm_templ.customdata.shape[:-1]

fig = go.Figure(
    data=[hm_templ],
    layout=go.Layout(
        # get xaxis from template heatmap
        xaxis=dict(autorange=True),
        yaxis=dict(autorange=True),
        title_text="Tokens visualized with the Tuned Lens",
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play", method="animate", args=[
                    None
                    # dict(frame=dict(duration=500, redraw=False), fromcurrent=True, mode="immediate"),
                ])
            ],
            direction="left",
            xanchor="right", yanchor="top", # required for right/top padding
            pad={"r": 10, "t": 100},
        )],
    ),
    frames=[
        go.Frame(
            data=[
                apply_metric('entropy', traj).heatmap(layer_stride=layer_stride)
            ],
            layout=go.Layout(
                title_text=f"Tokens visualized with the Tuned Lens, coeff={coeff}",
            ),
        )
        for coeff, traj in traj_list_by_coeff.items()
    ]
)
fig.show()

# %%

hm_templ.customdata.shape

# %%
# Plot kl divergence (coeff, next coeff) for each layer

def js_div(lp1, lp2):
    js_div = 0.5 * np.sum(
        lp1 * (lp1 - lp2), axis=-1
    ) + 0.5 * np.sum(lp2 * (lp2 - lp1), axis=-1)
    return np.log(1+js_div)

ys = np.array([
    js_div(
        traj_list_by_coeff[coeffs[i]].log_probs,
        traj_list_by_coeff[coeffs[i+1]].log_probs,
    )
    for i in range(len(coeffs)-1)
])


# Plot it

import plotly.graph_objects as go

fig = go.Figure()

for i in [48]: # range(ys.shape[1]):
    for j in range(ys.shape[2]):
        fig.add_trace(
            go.Scatter(
                x=coeffs[:-1],
                y=ys[:, i, j],
                mode="lines+markers",
                name=f"layer {i}, token {j}",
            )
        )

fig.update_layout(
    title="JS Divergence between adjacent coefficients",
    xaxis_title="Coefficient",
    yaxis_title="Log(1 + JS-Divergence)",
    legend_title="Layer, Token",
    # log both scales
    xaxis_type="log",
    # yaxis_type="log",
)
fig.show()



# %%
# Play with printing completions to check behavior


completion_utils.print_n_comparisons(
    prompt=prompt,
    num_comparisons=5,
    model=model,
    rich_prompts=rich_prompts,
    seed=0,
    temperature=1,
    # freq_penalty=1,
    top_p=0.8,
    tokens_to_generate=8,
)

# %%
