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


def run_corpus_loss_experiment(
    corpus_name: str,
    model: HookedTransformer,
    labeled_texts: pd.DataFrame,
    x_vector_phrases: Tuple[str, str],
    act_names: Union[str, np.ndarray],
    coeffs: Union[float, np.ndarray],
    method: str = "mask_injection_loss",
    text_col: str = "text",
    label_col: str = "label",
    x_qty: Optional[str] = "coeff",
    color_qty: Optional[str] = "label",
    facet_col_qty: Optional[str] = "act_name",
):
    """Function to evaluate loss on a set of input texts for both the
    original model and a model with various activation injections.  The
    injections are defined by a single pair of phrases and optional
    sweeps over coeff and act_name.  Results are presented over the
    classes present in the input text labels"""
    assert method in ["normal", "mask_injection_loss"], "Invalid method"
    # Create the metrics dict
    metrics_dict = {"loss": metrics.get_loss_metric(model, agg_mode=["full"])}
    # Get the loss on the original model
    normal_loss = metrics.add_metric_cols(
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
        normal_loss[["loss_full"]],
        on="text_index",
        lsuffix="_mod",
        rsuffix="_norm",
    )
    # Join in the RichPrompt parameters
    mod_df = mod_df.join(rich_prompts_df, on="rich_prompt_index")
    # Join in the text label
    mod_df = mod_df.join(labeled_texts[[label_col]], on="text_index")
    # Add loss diff column
    mod_df["loss_full_diff"] = (
        mod_df["loss_full_mod"] - mod_df["loss_full_norm"]
    )
    # Create a loss mean column, optionally masking out the loss at
    # positions that had activations injected
    if method == "mask_injection_loss":
        # NOTE: this assumes that the same phrases are used for all
        # RichPrompts, which is currently the case, but may not always be!
        mask_pos = rich_prompts_df.iloc[0]["rich_prompts"][0].tokens.shape[-1]
    else:
        mask_pos = 0
    mod_df["loss_mean_diff"] = mod_df["loss_full_diff"].apply(
        lambda inp: inp[mask_pos:].mean()
    )
    # Group results by label, coeff and act_name
    results_grouped_df = (
        mod_df.groupby(["act_name", "coeff", label_col])
        .mean(numeric_only=True)
        .reset_index()
    )
    # Plot the results
    return (
        px.line(
            results_grouped_df,
            y="loss_mean_diff",
            x=x_qty,
            color=color_qty,
            facet_col=facet_col_qty,
            title=f"Increase in mean loss for {corpus_name} over injection params, by {label_col}<br>"
            + f"method: {method}, phrases: "
            + f"{model.to_str_tokens(x_vector_phrases[0])} - "
            + f"{model.to_str_tokens(x_vector_phrases[1])}",
        ),
        mod_df,
        results_grouped_df,
    )
