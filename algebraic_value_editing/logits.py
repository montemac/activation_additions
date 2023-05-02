"""Functions for extracting and evaluating probability distributions
over next tokens with and without activation injections."""

from typing import List, Optional, Iterable, Union, Tuple, Any

import torch
import numpy as np
import pandas as pd
import plotly.express as px
from transformer_lens import HookedTransformer

from algebraic_value_editing import prompt_utils, hook_utils


def logits_to_probs_numpy(
    logits: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for converting logits to probs, returning
    both probabilities and logprobs as numpy arrays."""
    dist = torch.distributions.Categorical(logits=logits)
    return (
        dist.probs.detach().cpu().numpy(),
        dist.logits.detach().cpu().numpy(),
    )


def renorm_probs(probs: pd.DataFrame):
    """Convenience function to renormalize a distribution defined across
    DataFrame columns"""
    return probs.div(probs.sum(axis="columns"), axis="index")


def effectiveness(
    probs: pd.DataFrame,
    index: Any,
    is_steering_aligned: np.ndarray,
):
    """Function to calculate effectiveness given modified probabilities
    and logprob differences.  Also requires an is_steering_aligned
    boolnea array used to select which tokens to include in the
    calculation."""
    if not np.any(is_steering_aligned):
        return 0.0 * probs["mod", "probs"].loc[index, 0]
    return (
        probs["mod", "probs"].loc[index, is_steering_aligned]
        * (probs["mod", "logprobs"] - probs["normal", "logprobs"]).loc[
            index, is_steering_aligned
        ]
    ).sum(axis="columns")


def focus(
    probs: pd.DataFrame,
    index: Any,
    is_steering_aligned: np.ndarray,
):
    """Function calculate focus given normal and modified probabilities,
    and is_steering_aligned boolean array."""
    probs_norm_normed = renorm_probs(
        probs["normal", "probs"].loc[index, ~is_steering_aligned]
    )
    probs_mod_normed = renorm_probs(
        probs["mod", "probs"].loc[index, ~is_steering_aligned]
    )
    return (
        probs_mod_normed
        * (np.log(probs_mod_normed) - np.log(probs_norm_normed))
    ).sum(axis="columns")


def get_token_probs(
    model: HookedTransformer,
    prompts: Union[
        Union[str, torch.Tensor], Iterable[Union[str, torch.Tensor]]
    ],
    rich_prompts: Optional[List[prompt_utils.RichPrompt]] = None,
    return_positions_above: Optional[int] = None,
) -> pd.DataFrame:
    """Make a forward pass on a model for each provided prompted,
    optionally including hooks generated from RichPrompts provided.
    Return value is a DataFrame with tokens on the columns, prompts as
    index.
    """
    assert (
        return_positions_above is None
        or isinstance(prompts, str)
        or isinstance(prompts, torch.Tensor)
    ), "Can only return logits for multiple positions for a single prompt."
    # Add hooks if provided
    if rich_prompts is not None:
        hook_fns = hook_utils.hook_fns_from_rich_prompts(
            model=model,
            rich_prompts=rich_prompts,
        )
        for act_name, hook_fn in hook_fns.items():
            model.add_hook(act_name, hook_fn)
    # Try-except-finally to ensure hooks are cleaned up
    try:
        if return_positions_above is not None:
            tokens = model.to_tokens(prompts)[0, ...]
            probs_all, logprobs_all = logits_to_probs_numpy(
                model.forward(tokens)[0, return_positions_above:, :]
            )
            index = pd.Index(
                np.arange(return_positions_above, tokens.shape[-1]),
                name="pos",
            )
        else:
            probs_all = np.zeros((len(prompts), model.cfg.d_vocab_out))
            logprobs_all = np.zeros((len(prompts), model.cfg.d_vocab_out))
            for idx, prompt in enumerate(prompts):
                (
                    probs_all[idx, :],
                    logprobs_all[idx, :],
                ) = logits_to_probs_numpy(model.forward(prompt)[0, -1, :])
            index = pd.Index(prompts, name="prompt")
    except Exception as ex:
        raise ex
    finally:
        model.remove_all_hook_fns()
    # all_tokens = [
    #     model.tokenizer.decode(ii) for ii in range(model.cfg.d_vocab_out)
    # ]
    return pd.concat(
        (
            pd.DataFrame(data=probs_all, index=index),
            pd.DataFrame(data=logprobs_all, index=index),
        ),
        axis="columns",
        keys=["probs", "logprobs"],
    )


def get_for_tokens(
    inp: pd.DataFrame,
    tokens: np.ndarray,
    prepend_first_pos: Optional[Any] = None,
) -> np.ndarray:
    """Take values from an DataFrame where position and tokens are the index
    and columns, at the token positions provided.  The first element
    in tokens will be ignored, as is the case when e.g. taking probs for
    actual tokens in a sequence. An optional value can be prepended to
    ensure the returned array has the same position-dimension size."""
    inp_take = (
        np.take_along_axis(
            inp.values[:-1, :], tokens[1:, None], axis=-1
        ).squeeze(),
    )
    if prepend_first_pos is not None:
        inp_take = np.concatenate([[prepend_first_pos], inp_take])
    return inp_take


def get_normal_and_modified_token_probs(
    model: HookedTransformer,
    prompts: Iterable[str],
    rich_prompts: List[prompt_utils.RichPrompt],
    return_positions_above: Optional[int] = None,
) -> pd.DataFrame:
    """Get normal and modified next-token probabilities for a range of
    prompts, returning a DataFrame containing both"""
    normal_df = get_token_probs(
        model=model,
        prompts=prompts,
        return_positions_above=return_positions_above,
    )
    mod_df = get_token_probs(
        model=model,
        prompts=prompts,
        rich_prompts=rich_prompts,
        return_positions_above=return_positions_above,
    )
    return pd.concat(
        (normal_df, mod_df), axis="columns", keys=["normal", "mod"]
    )


def sort_tokens_by_probability(
    probs_df: pd.DataFrame,
):
    """Sort each token distribtion (normal, modified, all prompts)
    individually and return a single DF with rank as the index and the
    tokens and probs for each column as two-column pairs with a new
    column level."""
    # Sort by token probability in normal and modified, return as a
    # single df
    tokens_sorted_list = []
    for col in probs_df.columns:
        df_sorted = probs_df[col].sort_values(ascending=False)
        tokens_sorted_list.append(
            pd.DataFrame({"token": df_sorted.index, "prob": df_sorted.values})
        )
    return pd.concat(tokens_sorted_list, axis="columns", keys=probs_df.columns)


def plot_probs_changes(
    model: HookedTransformer, probs_df: pd.DataFrame, num: int = 30
):
    """Plot various view into token probabilities based on normal and
    modified distributions."""
    # Build up the DataFrame of probabilities with various other columns
    # for later px plotting.  Iterate over prompts and directions as
    # these will be the facet axes
    plot_datas = []
    for prompt in probs_df.columns.levels[1]:
        probs_df_this_prompt = probs_df.xs(prompt, axis="columns", level=1)
        prob_diff = (
            probs_df_this_prompt["mod"] - probs_df_this_prompt["normal"]
        )
        sort_ind_groups = {
            "increase": prob_diff.sort_values(ascending=False).iloc[:num],
            "decrease": prob_diff.sort_values(ascending=True).iloc[:num],
        }
        for direction, probs_sorted in sort_ind_groups.items():
            for qty, prob in {
                "normal prob": probs_df_this_prompt["normal"],
                "modified prob": probs_df_this_prompt["mod"],
                "mod-normal": prob_diff,
            }.items():
                plot_datas.append(
                    pd.DataFrame(
                        {
                            "prob": prob.loc[probs_sorted.index].values,
                            "token": probs_sorted.index,
                            "direction": direction,
                            "qty": qty,
                            "prompt": prompt,
                        }
                    )
                )
    # print(pd.concat(plot_datas))
    return px.line(
        pd.concat(plot_datas),
        y="prob",
        color="qty",
        facet_col="direction",
        facet_row="prompt",
        hover_data=["token"],
    )

    # # Org
    # prob_diff = probs_mod - probs_normal
    # sort_inds = np.argsort(prob_diff)
    # sort_ind_groups = {
    #     "increase": sort_inds[::-1][:num],
    #     "decrease": sort_inds[:num],
    # }
    # plot_datas = []
    # # Print tokens by prob_normal, prob_mod
    # print("by prob_normal, prob_mod")
    # dfs = []
    # for origin, probs in [("normal", probs_normal), ("mod", probs_mod)]:
    #     sort_inds_prob = np.argsort(probs)[::-1][:num]
    #     dfs.append(
    #         pd.DataFrame(
    #             {
    #                 f"token_{origin}": [
    #                     MODEL.tokenizer.decode(ind) for ind in sort_inds_prob
    #                 ],
    #                 f"prob_{origin}": probs[sort_inds_prob],
    #             }
    #         )
    #     )
    # print(pd.concat(dfs, axis="columns"))
    # for direction, inds in sort_ind_groups.items():
    #     tokens = [MODEL.tokenizer.decode(ind) for ind in inds]
    #     print(direction)
    #     print(pd.DataFrame({"token": tokens, "prob_diff": prob_diff[inds]}))
    #     for qty, prob in [
    #         ("normal prob", probs_normal),
    #         ("modified prob", probs_mod),
    #         ("mod-normal", prob_diff),
    #     ]:
    #         plot_datas.append(
    #             pd.DataFrame(
    #                 {
    #                     "prob": prob[inds],
    #                     "token": tokens,
    #                     "direction": direction,
    #                     "qty": qty,
    #                 }
    #             )
    #         )
    # px.line(
    #     pd.concat(plot_datas),
    #     y="prob",
    #     color="qty",
    #     facet_col="direction",
    #     hover_data=["token"],
    # ).show()
