"""Collection of wrapper functions for performing specific experiments,
which typically include some combination of data loading and processing,
analysis/sweeps/etc, and visualizing/summarizing results."""

from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly as py
import plotly.subplots

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    hook_utils,
    prompt_utils,
    utils,
    completion_utils,
    metrics,
    sweeps,
)


def run_corpus_logprob_experiment(
    corpus_name: str,
    model: HookedTransformer,
    labeled_texts: pd.DataFrame,
    x_vector_phrases: Tuple[str, str],
    act_names: Union[str, np.ndarray],
    coeffs: Union[float, np.ndarray],
    method: str = "mask_injection_logprob",
    text_col: str = "text",
    label_col: str = "label",
    x_qty: Optional[str] = "coeff",
    x_name: Optional[str] = None,
    color_qty: Optional[str] = "label",
    color_name: Optional[str] = None,
    facet_col_qty: Optional[str] = "act_name",
    facet_col_name: Optional[str] = None,
):
    """Function to evaluate log-prob on a set of input texts for both the
    original model and a model with various activation injections.  The
    injections are defined by a single pair of phrases and optional
    sweeps over coeff and act_name.  Results are presented over the
    classes present in the input text labels"""
    assert method in ["normal", "mask_injection_logprob"], "Invalid method"

    # Create pre and post forward functions so that we can use the KL
    # divergence metric with a hooked model

    # Create the metrics dict
    metrics_dict = {
        "logprob": metrics.get_logprob_metric(
            model,
            agg_mode=["actual_next_token", "kl_div"],
            q_model=model,
            q_funcs=(
                hook_utils.remove_and_return_hooks,
                hook_utils.add_hooks_from_dict,
            ),
        )
    }
    # Get the loss on the original model
    normal_metrics = metrics.add_metric_cols(
        labeled_texts[text_col].to_frame(),
        metrics_dict,
        cols_to_use=text_col,
        show_progress=True,
        prefix_cols=False,
    )
    # Create the list of RichPrompts based on provided hyperparameters
    rich_prompts_df = sweeps.make_rich_prompts(
        phrases=[[(x_vector_phrases[0], 1.0), (x_vector_phrases[1], -1.0)]],
        act_names=act_names,
        coeffs=coeffs,
        pad=True,
        model=model,
    )
    # Get the modified model losses over all the RichPrompts
    mod_df = sweeps.sweep_over_metrics(
        model=model,
        texts=labeled_texts[text_col],
        rich_prompts=rich_prompts_df["rich_prompts"],
        metrics_dict=metrics_dict,
        prefix_cols=False,
    )
    # Join the normal loss into the patched df so we can take diffs
    mod_df = mod_df.join(
        normal_metrics[["logprob_actual_next_token"]],
        on="text_index",
        lsuffix="_mod",
        rsuffix="_norm",
    )
    # Join in the RichPrompt parameters
    mod_df = mod_df.join(rich_prompts_df, on="rich_prompt_index")
    # Join in the text label
    mod_df = mod_df.join(labeled_texts[[label_col]], on="text_index")
    # Add loss diff column
    mod_df["logprob_actual_next_token_diff"] = (
        mod_df["logprob_actual_next_token_mod"]
        - mod_df["logprob_actual_next_token_norm"]
    )
    # Create a loss mean column, optionally masking out the loss at
    # positions that had activations injected
    if method == "mask_injection_logprob":
        # NOTE: this assumes that the same phrases are used for all
        # RichPrompts, which is currently the case, but may not always be!
        mask_pos = rich_prompts_df.iloc[0]["rich_prompts"][0].tokens.shape[-1]
    else:
        mask_pos = 0
    mod_df["logprob_actual_next_token_diff_mean"] = mod_df[
        "logprob_actual_next_token_diff"
    ].apply(lambda inp: inp[mask_pos:].mean())
    # Create a KL div mean column, also masking
    mod_df["logprob_kl_div_mean"] = mod_df["logprob_kl_div"].apply(
        lambda inp: inp[mask_pos:].mean()
    )
    # Group results by label, coeff and act_name
    results_grouped_df = (
        mod_df.groupby(["act_name", "coeff", label_col])
        .mean(numeric_only=True)
        .reset_index()
    )
    # Plot the results
    labels = {"logprob_actual_next_token_diff_mean": "Mean log-prob change"}
    if x_name is not None:
        labels[x_qty] = x_name
    if color_name is not None:
        labels[color_qty] = color_name
    if facet_col_name is not None:
        labels[facet_col_qty] = facet_col_name
    return (
        px.line(
            results_grouped_df,
            y="logprob_actual_next_token_diff_mean",
            x=x_qty,
            color=color_qty,
            facet_col=facet_col_qty,
            labels=labels,
            title=f"Increase in next-token log-prob for {corpus_name} over injection params, by {label_col}<br>"
            + f"method: {method}, phrases: "
            + f"{model.to_str_tokens(x_vector_phrases[0])} - "
            + f"{model.to_str_tokens(x_vector_phrases[1])}",
        ),
        mod_df,
        results_grouped_df,
    )
