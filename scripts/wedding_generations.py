"""Quick script to evaluating wedding-steered completions."""
# %%
# Imports, etc.
from typing import List

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import (
    completion_utils,
    utils,
    hook_utils,
    prompt_utils,
    metrics,
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)

# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to(
    "cuda:1"
)  # type: ignore


# %%
NUM_COMPLETIONS = 200

ACT_NAMES = np.arange(len(MODEL.blocks))

shared_params = dict(
    model=MODEL,
    prompt_batch=["I went up to my friend and said"] * NUM_COMPLETIONS,
    tokens_to_generate=40,
    seed=0,
    include_logits=False,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)
baseline_gens = metrics.add_metric_cols(
    completion_utils.gen_using_model(**shared_params), metrics_dict
)

actadd_gens_list = []
for act_name in tqdm(ACT_NAMES):
    activation_additions = list(
        prompt_utils.get_x_vector(
            prompt1=" weddings",
            prompt2="",
            coeff=1.0,
            act_name=int(act_name),
            model=MODEL,
            pad_method="tokens_right",
            custom_pad_id=MODEL.to_single_token(" "),  # type: ignore
        ),
    )

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

    actadd_gens = metrics.add_metric_cols(
        completion_utils.gen_using_activation_additions(
            activation_additions=activation_additions, **shared_params
        ),
        metrics_dict,
    )
    actadd_gens_list.append(actadd_gens)

# %%
# Analyze and plot results
results = []
for act_name, actadd_gens in zip(ACT_NAMES, actadd_gens_list):
    results.append(
        {
            "act_name": act_name,
            "actadd_mean": actadd_gens["wedding_words_count"].mean(),
            "actadd_nonzero": (actadd_gens["wedding_words_count"] > 0).mean(),
        }
    )
results_df = pd.DataFrame(results)
fig_mean = px.line(
    results_df,
    x="act_name",
    y="actadd_mean",
    labels={"actadd_mean": "Mean wedding word count", "act_name": "Layer"},
)
# Horizontal line with annotation positioned at left of plot
fig_mean.add_hline(
    y=baseline_gens["wedding_words_count"].mean(),
    line_dash="dot",
    annotation_text="baseline",
    annotation_position="top left",
)
fig_nonzero = px.line(
    results_df,
    x="act_name",
    y="actadd_nonzero",
    labels={
        "actadd_nonzero": "Non-zero wedding word count fraction",
        "act_name": "Layer",
    },
)
fig_nonzero.add_hline(
    y=baseline_gens["wedding_words_count"].mean(),
    line_dash="dot",
    annotation_text="baseline",
    annotation_position="top left",
)
figs = {"mean": fig_mean, "nonzero": fig_nonzero}
for name, fig in figs.items():
    utils.fig_to_publication_pdf(fig, f"images/wedding_gens_{name}")
