# visualization.py
# pyright: reportGeneralTypeIssues=false
import circuitsvis as cv
import torch

from typing import List, Dict

import streamlit.components.v1 as components
import streamlit as st
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import hook_utils
from activation_additions.prompt_utils import ActivationAddition


def plot_attention_pattern_single(
    tokens: List[int], attention: torch.Tensor
) -> None:
    """Plots the attention pattern for a single token."""
    assert len(tokens) == attention.shape[-1]

    result = cv.attention.attention_heads(
        tokens=tokens,
        attention=attention,
        max_value=1,
        min_value=-1,
    )
    # If there are more heads, we need to make the plot bigger
    components.html(
        str(result),
        width=400,
        height=600 + (100 * attention.shape[0] // 4),
    )


def attention_pattern_visualization(
    model: HookedTransformer,
    prompt_tokens: torch.Tensor,
    prompt_str_tokens: List[int],
    activation_adds: List[ActivationAddition],
) -> None:
    """Visualize the attention patterns before and after intervention."""

    hook_fns: Dict = hook_utils.hook_fns_from_activation_additions(
        model=model,
        activation_additions=activation_adds,
    )
    fwd_hooks = [
        (name, hook_fn)
        for name, hook_fns in hook_fns.items()
        for hook_fn in hook_fns
    ]

    st.subheader("Attention Pattern Visualization")

    attn_layer = st.slider(
        "Attention layer",
        min_value=0,
        max_value=model.cfg.n_layers - 1,
        value=0,
    )

    _, cache = model.run_with_cache(prompt_tokens, remove_batch_dim=True)
    attn_before = cache["pattern", attn_layer, "attn"]

    # Split visualization into two columns
    col1, col2, col3 = st.columns(3)

    # Visualize attention patterns before intervention
    with col1:
        st.subheader(f"Before intervention")
        plot_attention_pattern_single(
            tokens=prompt_str_tokens, attention=attn_before
        )

    # Perform intervention
    with model.hooks(fwd_hooks=fwd_hooks):  # type: ignore
        _, cache = model.run_with_cache(prompt_tokens, remove_batch_dim=True)
        attn_after = cache["pattern", attn_layer, "attn"]

    # Visualize attention patterns after intervention
    with col2:
        st.subheader(f"After intervention")
        plot_attention_pattern_single(
            tokens=prompt_str_tokens, attention=attn_after
        )

    # Compute difference in attention patterns
    attn_diff = attn_after - attn_before

    # Visualize the difference in attention patterns
    with col3:
        st.subheader(f"Difference")
        plot_attention_pattern_single(
            tokens=prompt_str_tokens, attention=attn_diff
        )


def attention_pattern_visualization():
    """Visualize the attention patterns before and after intervention."""
    model: HookedTransformer = st.session_state.model
    prompt_tokens: torch.Tensor = st.session_state.prompt_tokens
    prompt_str_tokens: List[int] = st.session_state.prompt_str_tokens

    hook_fns: Dict = hook_utils.hook_fns_from_activation_additions(
        model=model,
        activation_additions=st.session_state.activation_adds,
    )
    fwd_hooks = [
        (name, hook_fn)
        for name, hook_fns in hook_fns.items()
        for hook_fn in hook_fns
    ]

    attn_layer = st.slider(
        "Attention layer",
        min_value=0,
        max_value=model.cfg.n_layers - 1,
        value=0,
    )

    _, cache = model.run_with_cache(prompt_tokens, remove_batch_dim=True)
    attn_before = cache["pattern", attn_layer, "attn"]

    # Split visualization into two columns
    st.write(f"**Attention patterns for layer {attn_layer}**")
    col1, col2, col3 = st.columns(3)

    # Visualize attention patterns before intervention
    with col1:
        st.subheader(f"Before intervention")
        plot_attention_pattern_single(
            tokens=prompt_str_tokens, attention=attn_before
        )

    # Perform intervention
    with model.hooks(fwd_hooks=fwd_hooks):  # type: ignore
        _, cache = model.run_with_cache(prompt_tokens, remove_batch_dim=True)
        attn_after = cache["pattern", attn_layer, "attn"]

    # Visualize attention patterns after intervention
    with col2:
        st.subheader(f"After intervention")
        plot_attention_pattern_single(
            tokens=prompt_str_tokens, attention=attn_after
        )

    # Compute difference in attention patterns
    attn_diff = attn_after - attn_before

    # Visualize the difference in attention patterns
    with col3:
        st.subheader(f"Difference")
        plot_attention_pattern_single(
            tokens=prompt_str_tokens, attention=attn_diff
        )
