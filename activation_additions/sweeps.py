""" Functions for performing automated sweeps of algebraic value editing
over layers, coeffs, etc. """

from typing import Iterable, Optional, List, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from activation_additions import metrics, logging, hook_utils
from activation_additions.prompt_utils import ActivationAddition
from activation_additions.completion_utils import (
    gen_using_hooks,
    gen_using_activation_additions,
)


@logging.loggable
def make_activation_additions(
    phrases: List[List[Tuple[str, float]]],
    act_names: Union[List[str], List[int], np.ndarray],
    coeffs: Union[List[float], np.ndarray],
    pad: bool = False,
    model: Optional[HookedTransformer] = None,
    log: Union[bool, Dict] = False,  # pylint: disable=unused-argument
) -> pd.DataFrame:
    """Make a single series of ActivationAddition lists by combining all permutations
    of lists of phrases with initial coeffs, activation names (i.e. layers), and
    additional coeffs that are applied to lists of phrases as an
    additional scale factor. For example, a 'phrase diff' at various coeffs can
    be created by passing `phrases=[[(phrase1, 1.0), (phrase2, -1.0)]]`
    and `coeffs=[-10, -1, 1, 10]`.  Phrases can optionally be padded so
    that each ActivationAddition that will be injected simultaneously has the
    same token length.  Padding uses spaces and is done at right (this
    might be changed in future to allow the same args as the x_vector
    function). If pad==True, model must be provided also.

    The returned DataFrame has columns for the ActivationAddition lists, and the
    inputs that generated each one (i.e. phrases, act_name, coeff)."""
    assert (
        pad is False or model is not None
    ), "model must be provided if pad==True"
    rows = []
    for phrases_this in phrases:
        for act_name in act_names:
            for coeff in coeffs:
                activation_additions_this = []
                if pad:
                    pad_token: int = model.to_single_token(" ")  # type: ignore
                    # Convert all phrases into tokens
                    tokens_list = [
                        model.to_tokens(phrase)[0]  # type: ignore
                        for phrase, init_coeff in phrases_this
                    ]
                    # Get max length of tokens
                    max_len = max(tokens.shape[-1] for tokens in tokens_list)
                    # Pad each phrase's tokens
                    tokens_list = [
                        torch.nn.functional.pad(
                            tokens,
                            (0, max_len - tokens.shape[-1]),
                            mode="constant",
                            value=pad_token,
                        )
                        for tokens in tokens_list
                    ]
                    # Create the ActivationAdditions using the padded tokens
                    for (phrase, init_coeff), tokens in zip(
                        phrases_this, tokens_list
                    ):
                        activation_additions_this.append(
                            ActivationAddition(
                                coeff=init_coeff * coeff,
                                act_name=act_name,
                                tokens=tokens.squeeze(),
                            )
                        )
                else:
                    # Create the ActivationAdditions using the phrase strings
                    for phrase, init_coeff in phrases_this:
                        activation_additions_this.append(
                            ActivationAddition(
                                coeff=init_coeff * coeff,
                                act_name=act_name,
                                prompt=phrase,
                            )
                        )
                rows.append(
                    {
                        "activation_additions": activation_additions_this,
                        "phrases": phrases_this,
                        "act_name": act_name,
                        "coeff": coeff,
                    }
                )

    return pd.DataFrame(rows)


@logging.loggable
def sweep_over_prompts(
    model: HookedTransformer,
    prompts: Iterable[str],
    activation_additions: Iterable[List[ActivationAddition]],
    num_normal_completions: int = 100,
    num_patched_completions: int = 100,
    tokens_to_generate: int = 40,
    seed: Optional[int] = None,
    metrics_dict: Optional[
        Union[
            Dict[str, metrics.TextMetricFunc],
            Dict[str, metrics.TokensMetricFunc],
        ]
    ] = None,
    log: Union[bool, Dict] = False,  # pylint: disable=unused-argument
    **sampling_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply each provided ActivationAddition to each prompt num_completions
    times, returning the results in a dataframe.  The iterable of
    ActivationAdditions may be created directly for simple cases, or created by
    sweeping over e.g. layers, coeffs, ingredients, etc. using other
    functions in this module.

    args:
        model: The model to use for completion.

        prompts: The prompts to use for completion.

        activation_additions: An iterable of ActivationAddition lists to patch into the
        prompts, in all permutations.

        num_normal_completions: Number of completions to generate for each
        prompt for the normal, unpatched model.

        num_patched_completions: Number of completions to generate for each
        prompt/ActivationAddition combination.

        tokens_to_generate: The number of additional tokens to generate.

        seed: A random seed to use for generation.

        `log`: To enable logging of this call to wandb, pass either
        True, or a dict contining any of ('tags', 'group', 'notes') to
        pass these keys to the wandb init call.  False to disable logging.

        sampling_kwargs: Keyword arguments to pass to the model's
        generate function.

    returns:
        A tuple of DataFrames, one containing normal, unpatched
        completions for each prompt, the other containing patched
        completions.
    """
    # Iterate over prompts
    normal_list = []
    patched_list = []
    for prompt in tqdm(prompts):
        # Generate the normal completions for this prompt, with logging
        # forced off since we'll be logging to final DataFrames
        normal_df: pd.DataFrame = gen_using_hooks(
            model=model,
            prompt_batch=[prompt] * num_normal_completions,
            hook_fns={},
            tokens_to_generate=tokens_to_generate,
            seed=seed,
            log=False,
            **sampling_kwargs,
        )
        # Append for later concatenation
        normal_list.append(normal_df)
        # Iterate over ActivationAdditions
        for index, activation_additions_this in enumerate(
            tqdm(activation_additions)
        ):
            # Generate the patched completions, with logging
            # forced off since we'll be logging to final DataFrames
            patched_df: pd.DataFrame = gen_using_activation_additions(
                model=model,
                activation_additions=activation_additions_this,
                prompt_batch=[prompt] * num_patched_completions,
                tokens_to_generate=tokens_to_generate,
                seed=seed,
                log=False,
                **sampling_kwargs,
            )
            patched_df["activation_addition_index"] = index
            # Store for later
            patched_list.append(patched_df)
    # Create the final normal and patched completion frames
    normal_all = pd.concat(normal_list).reset_index(names="completion_index")
    patched_all = pd.concat(patched_list).reset_index(names="completion_index")
    # Create and add metric columns
    if metrics_dict is not None:
        normal_all = metrics.add_metric_cols(normal_all, metrics_dict)
        patched_all = metrics.add_metric_cols(patched_all, metrics_dict)
    return normal_all, patched_all


# TODO: this interface overall is somewhat awkward and could be
# re-designed.  In general it might make sense to redesign with more
# than just model completions in mind?
@logging.loggable
def sweep_over_metrics(
    model: HookedTransformer,
    inputs: Union[Iterable[Any], pd.Series],
    activation_additions: Iterable[List[ActivationAddition]],
    metrics_dict: Union[
        Dict[str, metrics.TextMetricFunc],
        Dict[str, metrics.TokensMetricFunc],
    ],
    log: Union[bool, Dict] = False,  # pylint: disable=unused-argument
    **metric_args,
) -> pd.DataFrame:
    """Apply all the metrics to the provided input texts after hooking the
    provided model with each of the provided ActivationAdditions in turn.  The
    iterable of ActivationAdditions may be created directly for simple cases, or
    created by sweeping over e.g. layers, coeffs, ingredients, etc.
    using other functions in this module.

    The expected use case for this function is that the metrics in
    metrics_dict are closed over the same provided model, so that
    changing the model via hooking will change the output of the metric
    functions.  This design may change in future.

    args:
        model: The model to apply ActivationAdditions to.

        inputs: The inputs to apply the metrics to. Can be an
        iterable or a Series.

        activation_additions: An iterable of ActivationAddition lists to patch into the
        model.

        metrics_dict: A dict of named metric functions.

        log: To enable logging of this call to wandb, pass either
        True, or a dict contining any of ('tags', 'group', 'notes') to
        pass these keys to the wandb init call.  False to disable logging.

    returns:
        A DataFrame of metric outputs
    """
    # Create the input text DataFrame
    inputs_df = pd.DataFrame({"input": inputs})
    # Iterate over ActivationAdditions
    patched_list = []
    for index, activation_additions_this in enumerate(
        tqdm(activation_additions)
    ):
        hook_fns = hook_utils.hook_fns_from_activation_additions(
            model=model,
            activation_additions=activation_additions_this,
        )
        # Get the modified loss and append
        model.remove_all_hook_fns()
        for act_name, hook_fns_list in hook_fns.items():
            for hook_fn in hook_fns_list:
                model.add_hook(act_name, hook_fn)
        patched_df = metrics.add_metric_cols(
            inputs_df, metrics_dict, cols_to_use="input", **metric_args
        )
        patched_df["activation_addition_index"] = index
        patched_list.append(patched_df)
        model.remove_all_hook_fns()

    # Create the final patched df and return both
    patched_all = pd.concat(patched_list).reset_index(names="input_index")
    return patched_all


def reduce_sweep_results(
    normal_df: pd.DataFrame,
    patched_df: pd.DataFrame,
    activation_additions_df: pd.DataFrame,
):
    """Perform some common post-processing on sweep results to:
    - take means for all metrics over all repititions of each
      (ActivationAddition, prompt) pair,
    - join ActivationAddition information into patched data so that phrases,
      coeffs, act_names are available as columns"""
    reduced_df = patched_df.groupby(
        ["prompts", "activation_addition_index"]
    ).mean(numeric_only=True)
    reduced_joined_df = reduced_df.join(
        activation_additions_df, on="activation_addition_index"
    ).reset_index()
    reduced_normal_df = normal_df.groupby(["prompts"]).mean(numeric_only=True)
    return reduced_normal_df, reduced_joined_df


def plot_sweep_results(
    data: pd.DataFrame,
    col_to_plot: str,
    title: str,
    col_x="coeff",
    col_color="act_name",
    col_facet_col="prompts",
    col_facet_row=None,
    baseline_data=None,
    px_func=px.line,
) -> go.Figure:
    """Plot the reduced results of a sweep, with controllable axes,
    colors, etc.  Pass a reduced normal-completions DataFrame into
    `baseline_data` to add horizontal lines for metric baselines."""
    fig: go.Figure = px_func(
        data,
        title=title,
        color=col_color,
        y=col_to_plot,
        x=col_x,
        facet_col=col_facet_col,
        facet_row=col_facet_row,
    )
    # TODO: generalize this to any facet row/col config
    if (
        baseline_data is not None
        and col_to_plot in baseline_data
        and col_facet_col == "prompts"
        and col_facet_row is None
    ):
        for index, prompt in enumerate(baseline_data.index):
            fig.add_hline(
                y=baseline_data.loc[prompt][col_to_plot],
                row=1,  # type: ignore
                col=index + 1,  # type: ignore because int is valid for row/col
                annotation_text="normal",
                annotation_position="bottom left",
            )
    return fig
