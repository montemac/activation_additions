# Streamlit app for exploring activation additions
import torch
from typing import List, Dict
import transformer_lens

from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import hook_utils, prompt_utils, completion_utils
from algebraic_value_editing.prompt_utils import ActivationAddition

import streamlit as st
from streamlit.components.v1 import html
import circuitsvis as cv


DEVICE: str = "cuda"  # Default device
DEFAULT_KWARGS: Dict = {
    "seed": 0,
    "temperature": 1.0,
    "freq_penalty": 1.0,
    "top_p": 0.3,
    "num_comparisons": 15,
    # "logging": {"tags": ["linear prompt combo"]},
}


@st.cache_data
def load_model_tl(model_name: str, device: str = "cpu") -> HookedTransformer:
    """Loads a model on CPU and then transfers it to the device."""
    model: HookedTransformer = HookedTransformer.from_pretrained(
        model_name, device="cpu"
    )
    _ = model.to(device)
    return model


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
    html(
        str(result),
        width=400,
        height=600 + (100 * attention.shape[0] // 4),
    )


# Save memory by not computing gradients
_ = torch.set_grad_enabled(False)
torch.manual_seed(0)  # For reproducibility

gpt2small: HookedTransformer = load_model_tl(
    model_name="gpt2-small", device=DEVICE
)


def main():
    st.title("Algebraic Value Editing Demo")

    # User inputs
    with st.sidebar:
        model_name: str = st.selectbox(
            "Select GPT-2 Model",
            ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        )
        model: HookedTransformer = load_model_tl(
            model_name=model_name, device=DEVICE
        )
        prompt: str = st.text_input(
            "Prompt", value="My name is Frank and I like to eat"
        )

        act_prompt_1 = st.text_input("Act add prompt 1", value="Love")
        act_prompt_2 = st.text_input("Act add prompt 2", value="Hate")
        addition_layer: int = st.slider(
            "Injection site",
            min_value=0,
            max_value=model.cfg.n_layers,
            value=0,
        )
        coefficient: float = st.number_input("Coefficient", value=1.0)

    # Convert sample text to tokens
    sample_tokens = model.to_tokens(prompt)
    sample_str_tokens = model.to_str_tokens(prompt)

    # Get hooks for the activation addition on the GPT-2 model
    activation_adds: List[ActivationAddition] = [
        *prompt_utils.get_x_vector(
            act_prompt_1, act_prompt_2, coefficient, addition_layer
        )
    ]
    hook_fns: Dict = hook_utils.hook_fns_from_activation_additions(
        model=model,
        activation_additions=activation_adds,
    )
    fwd_hooks = [
        (name, hook_fn)
        for name, hook_fns in hook_fns.items()
        for hook_fn in hook_fns
    ]

    attn_layer: int = st.slider(
        "Attention layer",
        min_value=0,
        max_value=model.cfg.n_layers,
        value=0,
    )

    logits, cache = model.run_with_cache(sample_tokens, remove_batch_dim=True)
    attn_before = cache["pattern", attn_layer, "attn"]

    # Split visualization into two columns
    st.header(f"Attention patterns for layer {attn_layer}")
    col1, col2 = st.columns(2)

    # Visualize attention patterns before intervention
    with col1:
        st.subheader(f"Before intervention")
        plot_attention_pattern_single(
            tokens=sample_str_tokens, attention=attn_before
        )

    # Perform intervention
    with model.hooks(fwd_hooks=fwd_hooks):
        logits, cache = model.run_with_cache(
            sample_tokens, remove_batch_dim=True
        )
        attn_after = cache["pattern", attn_layer, "attn"]

    # Visualize attention patterns after intervention
    with col2:
        st.subheader(f"After intervention")
        plot_attention_pattern_single(
            tokens=sample_str_tokens, attention=attn_after
        )

    # Compute difference in attention patterns
    attn_diff = attn_after - attn_before

    # Visualize the difference in attention patterns
    st.subheader(f"Difference")
    plot_attention_pattern_single(
        tokens=sample_str_tokens, attention=attn_diff
    )


if __name__ == "__main__":
    main()
