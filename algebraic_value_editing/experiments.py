"""Collection of wrapper functions for performing specific experiments,
which typically include some combination of data loading and processing,
analysis/sweeps/etc, and visualizing/summarizing results."""

from typing import Tuple, Union, Optional, List

import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    prompt_utils,
    metrics,
    sweeps,
    logits,
)


def run_corpus_logprob_experiment(
    model: HookedTransformer,
    labeled_texts: pd.DataFrame,
    x_vector_phrases: Tuple[str, str],
    act_names: Union[str, np.ndarray],
    coeffs: Union[float, np.ndarray],
    method: str = "mask_injection_logprob",
    text_col: str = "text",
    label_col: str = "label",
):
    """Function to evaluate log-prob on a set of input texts for both the
    original model and a model with various activation injections.  The
    injections are defined by a single pair of phrases and optional
    sweeps over coeff and act_name.  Results are presented over the
    classes present in the input text labels"""
    assert method in [
        "normal",
        "mask_injection_logprob",
        "pad",
    ], "Invalid method"

    # Create the metrics dict
    metrics_dict = {
        "logprob": metrics.get_logprob_metric(
            model,
            # agg_mode=["actual_next_token", "kl_div"],
            agg_mode=["actual_next_token"],
            q_model=model,
            # q_funcs=(
            #     hook_utils.remove_and_return_hooks,
            #     hook_utils.add_hooks_from_dict,
            # ),
        )
    }
    # Create the list of RichPrompts based on provided hyperparameters
    rich_prompts_df = sweeps.make_rich_prompts(
        phrases=[[(x_vector_phrases[0], 1.0), (x_vector_phrases[1], -1.0)]],
        act_names=act_names,
        coeffs=coeffs,
        pad=True,
        model=model,
    )
    # Create the texts to use, optinally including padding
    tokens_list = [model.to_tokens(text) for text in labeled_texts[text_col]]
    if method == "pad":
        rich_prompts_all = []
        for rich_prompts in rich_prompts_df["rich_prompts"]:
            rich_prompts_all.extend(rich_prompts)
        tokens_list = [
            prompt_utils.pad_tokens_to_match_rich_prompts(
                model=model, tokens=tokens, rich_prompts=rich_prompts_all
            )[0]
            for tokens in tokens_list
        ]
    # Hack to avoid Pandas from trying to parse out the tokens tensors
    tokens_df = pd.DataFrame.from_records(
        [(tokens,) for tokens in tokens_list], index=labeled_texts.index
    ).rename({0: "tokens"}, axis="columns")
    # Get the logprobs on the original model
    normal_metrics = metrics.add_metric_cols(
        tokens_df,
        metrics_dict,
        cols_to_use="tokens",
        show_progress=True,
        prefix_cols=False,
    )
    # Get the modified model logprobs over all the RichPrompts
    mod_df = sweeps.sweep_over_metrics(
        model=model,
        inputs=tokens_df["tokens"],  # pylint: disable=unsubscriptable-object
        rich_prompts=rich_prompts_df["rich_prompts"],
        metrics_dict=metrics_dict,
        prefix_cols=False,
    )
    # Join the normal logprobs into the patched df so we can take diffs
    mod_df = mod_df.join(
        normal_metrics[["logprob_actual_next_token"]],
        on="input_index",
        lsuffix="_mod",
        rsuffix="_norm",
    )
    # Join in the RichPrompt parameters
    mod_df = mod_df.join(rich_prompts_df, on="rich_prompt_index")
    # Join in the input label
    mod_df = mod_df.join(labeled_texts[[label_col]], on="input_index")
    # Add loss diff column
    mod_df["logprob_actual_next_token_diff"] = (
        mod_df["logprob_actual_next_token_mod"]
        - mod_df["logprob_actual_next_token_norm"]
    )
    # Create a loss mean column, optionally masking out the loss at
    # positions that had activations injected
    if method in ["mask_injection_logprob", "pad"]:
        # NOTE: this assumes that the same phrases are used for all
        # RichPrompts, which is currently the case, but may not always be!
        mask_pos = rich_prompts_df.iloc[0]["rich_prompts"][0].tokens.shape[-1]
    else:
        mask_pos = 0
    mod_df["logprob_actual_next_token_diff_mean"] = mod_df[
        "logprob_actual_next_token_diff"
    ].apply(lambda inp: inp[mask_pos:].mean())
    # Create a KL div mean column, also masking
    # mod_df["logprob_kl_div_mean"] = mod_df["logprob_kl_div"].apply(
    #     lambda inp: inp[mask_pos:].mean()
    # )
    # Group results by label, coeff and act_name
    results_grouped_df = (
        mod_df.groupby(["act_name", "coeff", label_col])
        .mean(numeric_only=True)
        .reset_index()
    )
    # Return the results
    return mod_df, results_grouped_df


def plot_corpus_logprob_experiment(
    results_grouped_df: pd.DataFrame,
    corpus_name: str,
    x_qty: Optional[str] = "coeff",
    x_name: Optional[str] = None,
    color_qty: Optional[str] = "label",
    color_name: Optional[str] = None,
    facet_col_qty: Optional[str] = "act_name",
    facet_col_name: Optional[str] = None,
    **plot_kwargs,
):
    """Plot the results of a previously run corpus experiment"""
    labels = {
        "logprob_actual_next_token_diff_mean": "Mean change in log-probs"
    }
    if x_name is not None:
        labels[x_qty] = x_name
    if color_name is not None:
        labels[color_qty] = color_name
    if facet_col_name is not None:
        labels[facet_col_qty] = facet_col_name
    fig = px.line(
        results_grouped_df,
        y="logprob_actual_next_token_diff_mean",
        x=x_qty,
        color=color_qty,
        facet_col=facet_col_qty,
        labels=labels,
        title=f"Average change in log-probabilities of tokens in {corpus_name}",
        **plot_kwargs,
    )
    for annot in fig.layout.annotations:
        if "=" in annot.text:
            annot.update(
                text=" ".join(annot.text.split("=")), font={"size": 16}
            )
    return fig


def show_token_probs(
    model: HookedTransformer,
    probs_norm: Union[np.ndarray, pd.DataFrame],
    probs_mod: Union[np.ndarray, pd.DataFrame],
    pos: int,
    top_k: int,
    sort_mode: str = "prob",
    extra_title: str = "",
    token_strs_to_ignore: Union[list, np.ndarray] = None,
):
    """Print probability changes of top-K tokens for a specific input
    sequence, sorted using a specific sorting mode.

    Arguments probs_norm and probs_mod can either be ndarrays, or
    DataFrames; in either case, the dimensions must be (position, token)."""
    assert sort_mode in ["prob", "kl_div"]
    # Pick out the provided position for convenience
    if isinstance(probs_norm, pd.DataFrame):
        probs_norm = probs_norm.values
    if isinstance(probs_mod, pd.DataFrame):
        probs_mod = probs_mod.values
    probs_norm = probs_norm[pos, :]
    probs_mod = probs_mod[pos, :]
    # Set probs to zero and renormalize for tokens to ignore
    keep_mask = np.ones_like(probs_norm, dtype=bool)
    if token_strs_to_ignore is not None:
        tokens_to_ignore = np.array(
            [
                model.to_single_token(token_str)
                for token_str in token_strs_to_ignore
            ]
        )
        keep_mask[tokens_to_ignore] = False
        probs_norm[~keep_mask] = 0.0
        probs_norm /= probs_norm[keep_mask].sum()
        probs_mod[~keep_mask] = 0.0
        probs_mod /= probs_mod[keep_mask].sum()
    # Sort
    if sort_mode == "prob":
        norm_top_k = np.argsort(probs_norm)[::-1][:top_k]
        mod_top_k = np.argsort(probs_mod)[::-1][:top_k]
        top_k_tokens = np.array(list(set(norm_top_k).union(set(mod_top_k))))
        y_values = probs_mod[top_k_tokens] / probs_norm[top_k_tokens]
        y_title = "Modified/normal token probability ratio"
        title = (
            f"Probability ratio vs normal-model probabilities {extra_title}"
        )
    elif sort_mode == "kl_div":
        kl_contrib = np.ones_like(probs_mod)
        kl_contrib[keep_mask] = probs_mod[keep_mask] * np.log(
            probs_mod[keep_mask] / probs_norm[keep_mask]
        )
        top_k_tokens = np.argsort(kl_contrib)[::-1][
            :top_k
        ].copy()  # Copy to avoid negative stride
        y_values = kl_contrib[top_k_tokens]
        y_title = "Contribution to KL divergence (nats)"
        title = f"Contribution to KL divergence vs normal-model probabilities {extra_title}"

    plot_df = pd.DataFrame(
        {
            "probs_norm": probs_norm[top_k_tokens],
            "y_values": y_values,
            "text": model.to_string(top_k_tokens[:, None]),
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["probs_norm"],
            y=plot_df["y_values"],
            text=plot_df["text"],
            textposition="top center",
            mode="markers+text",
            marker_color=px.colors.qualitative.Plotly[0],
            showlegend=False,
        )
    )
    if sort_mode == "prob":
        min_prob = plot_df["probs_norm"].values.min()
        max_prob = plot_df["probs_norm"].values.max()
        unit_line_x = np.array([min_prob, max_prob])
        unit_line_y = np.array([1, 1])
        fig.add_trace(
            go.Scatter(
                x=unit_line_x,
                y=unit_line_y,
                mode="lines",
                line={"dash": "dot"},
                name="modified = normal",
                line_color=px.colors.qualitative.Plotly[1],
                showlegend=False,
            )
        )
    # Figure tweaking
    if sort_mode == "prob":
        fig.update_yaxes(type="log")
    fig.update_xaxes(type="log")
    fig.update_layout(
        title_text=title,
        xaxis_title="Normal model token probability",
        yaxis_title=y_title,
    )
    fig.update_traces(textposition="top center")
    return fig, plot_df


def compare_with_prompting(
    model: HookedTransformer,
    text: str,
    phrases: Tuple[str, str],
    coeff: float,
    act_names: List[Union[int, str]],
    pos: int,
):
    """Compare activation-injection at specified layers with prompting,
    using a space-padded input to make the techniques as directly
    comparable as possible."""
    probs_dict = {}

    # Normal
    probs_normal = logits.get_token_probs(
        model=model,
        prompts=text,
        return_positions_above=0,
    )
    len_normal = probs_normal.shape[0]
    tokens_str_normal = model.to_str_tokens(text)

    tokens_padded = model.to_tokens(text, prepend_bos=False)
    text_tokens_len = tokens_padded.shape[-1]
    rich_prompt_tokens_len = model.to_tokens(
        phrases[0], prepend_bos=False
    ).shape[-1]
    while tokens_padded.shape[-1] < text_tokens_len + rich_prompt_tokens_len:
        tokens_padded = torch.concat(
            (model.to_tokens(" ", prepend_bos=False), tokens_padded), axis=-1
        )
    tokens_padded = torch.concat((model.to_tokens(""), tokens_padded), axis=-1)

    # Prompted
    tokens_prompted = torch.concat(
        (
            model.to_tokens(phrases[0]),
            model.to_tokens(text, prepend_bos=False),
        ),
        axis=1,
    )
    probs_dict["prompted"] = (
        logits.get_token_probs(
            model=model,
            prompts=tokens_prompted,
            return_positions_above=0,
        )
        .iloc[-len_normal:]
        .reset_index(drop=True)
    )

    # Injected, space-padded, different layers
    for act_name in act_names:
        name = f"layer {act_name}" if isinstance(act_name, int) else act_name
        probs_dict[name] = (
            logits.get_token_probs(
                model=model,
                prompts=tokens_padded,
                return_positions_above=0,
                # pylint: disable=duplicate-code
                rich_prompts=list(
                    prompt_utils.get_x_vector(
                        prompt1=phrases[0],
                        prompt2=phrases[1],
                        coeff=coeff,
                        act_name=act_name,
                        model=model,
                        pad_method="tokens_right",
                        custom_pad_id=model.to_single_token(" "),
                    )
                ),
                # pylint: enable=duplicate-code
            )
            .iloc[-len_normal:]
            .reset_index(drop=True)
        )

    # Compare them all to the normal probs
    figs_dict = {}
    fig = go.Figure()
    for name, probs in probs_dict.items():
        kl_div = (
            probs["probs"] * (probs["logprobs"] - probs_normal["logprobs"])
        ).sum(axis="columns")
        fig.add_trace(
            go.Scatter(
                x=[
                    f"{pp}: {tok_str}"
                    for pp, tok_str in enumerate(tokens_str_normal)
                    if pp >= 1
                ],
                y=kl_div.iloc[1:],
                name=name,
            )
        )
    fig.update_layout(
        title_text="KL divergence over input for different steering methods"
    )
    figs_dict["kl"] = fig

    if pos is None:
        pos = probs_normal.shape[0] - 1

    def show_by_name(name):
        fig, _ = show_token_probs(
            model,
            probs_normal["probs"].values,
            probs_dict[name]["probs"].values,
            pos,
            10,
            sort_mode="kl_div",
            extra_title=f'<br>Input: "{"".join(tokens_str_normal[1 : (pos + 1)])}", method: {name}',
        )
        figs_dict[name] = fig

    for name in probs_dict:
        show_by_name(name)

    return figs_dict
