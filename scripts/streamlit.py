# Streamlit app for exploring activation additions
import io
import sys
from typing import List, Dict

import torch
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import hook_utils, prompt_utils, completion_utils
from activation_additions.prompt_utils import ActivationAddition

import streamlit as st
from streamlit.components.v1 import html
import circuitsvis as cv


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


def customize_activation_addition():
    st.sidebar.title("Model, prompt, and activation additions")

    model_name = st.selectbox(
        "Select GPT-2 Model",
        ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    )
    # Load the GPT-2 model
    model = load_model_tl(model_name=model_name, device="cuda")
    st.session_state.model = model

    prompt: str = st.sidebar.text_input(
        "Prompt", value="My name is Frank and I like to eat"
    )
    st.session_state.prompt = prompt

    st.markdown("<hr>", unsafe_allow_html=True)

    act_prompt_1 = st.sidebar.text_input("Act add prompt 1", value="Love")
    act_prompt_2 = st.sidebar.text_input("Act add prompt 2", value="Hate")
    addition_layer: int = st.sidebar.slider(
        "Injection site",
        min_value=0,
        max_value=model.cfg.n_layers - 1,
        value=0,
    )
    st.session_state.coefficient = st.sidebar.number_input(
        "Coefficient", value=1.0
    )

    # Convert sample text to tokens
    st.session_state.prompt_tokens = model.to_tokens(prompt)
    st.session_state.prompt_str_tokens = model.to_str_tokens(prompt)

    # Get hooks for the activation addition on the GPT-2 model
    activation_adds: List[ActivationAddition] = [
        *prompt_utils.get_x_vector(
            act_prompt_1,
            act_prompt_2,
            st.session_state.coefficient,
            addition_layer,
        )
    ]
    st.session_state.activation_adds = activation_adds


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
    st.header(f"Attention patterns for layer {attn_layer}")
    col1, col2, col3 = st.columns(3)

    # Visualize attention patterns before intervention
    with col1:
        st.subheader(f"Before intervention")
        plot_attention_pattern_single(
            tokens=prompt_str_tokens, attention=attn_before
        )

    # Perform intervention
    with model.hooks(fwd_hooks=fwd_hooks):
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


def completion_generation():
    """Provides tools for running completions."""
    # Let user configure non-negative temperature and frequency penalty and top_p and
    # integer num_comparisons and seed
    temperature = st.slider("Temperature", min_value=0.0, value=1.0)
    freq_penalty = st.slider(
        "Frequency penalty", min_value=0.0, max_value=2.0, value=1.0
    )
    top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.3)
    num_comparisons = st.slider(
        "Number of completions", min_value=1, value=5, step=1
    )
    seed = st.number_input("Random seed", value=0, step=1)
    tokens_to_generate = st.number_input(
        "Tokens to generate", min_value=0, value=50, step=1
    )

    # Create a "loading" placeholder
    placeholder = st.empty()
    placeholder.write("Loading...")

    # Redirect stdout to a StringIO object
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture

    completion_utils.print_n_comparisons(
        model=st.session_state.model,
        activation_additions=st.session_state.activation_adds,
        prompt=st.session_state.prompt,
        num_comparisons=num_comparisons,
        tokens_to_generate=tokens_to_generate,
        temperature=temperature,
        freq_penalty=freq_penalty,
        top_p=top_p,
        seed=seed,
    )

    # Retrieve the captured stdout
    completions_output = stdout_capture.getvalue()
    # Remove ANSI escape sequences (previously, bold formatting)
    completions_output = completions_output.replace("[1m", "")
    completions_output = completions_output.replace("[0m", "")

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Display the completions in the Streamlit app
    st.code(completions_output, language=None)

    # Remove the loading indicator
    placeholder.empty()


def next_token_stats() -> None:
    """Plot the next token probabilities, KL(). Writes output to
    streamlit."""
    # Calculate normal and modified token probabilities
    from activation_additions import logits, experiments
    import plotly.graph_objects as go
    import pandas as pd

    probs: pd.DataFrame = logits.get_normal_and_modified_token_probs(
        model=st.session_state.model,
        prompts=st.session_state.prompt,
        activation_additions=st.session_state.activation_adds,
        return_positions_above=0,  # NOTE idk what this does
    )

    # Show token probabilities figure
    top_k: int = 10
    fig, _ = experiments.show_token_probs(
        st.session_state.model,
        probs["normal", "probs"],
        probs["mod", "probs"],
        -1,
        top_k,
    )

    # Adjusting figure layout
    fig.update_layout(width=500)
    st.write(fig)

    # Calculate KL divergence and entropy
    kl_divergence: float = (
        (
            probs["mod", "probs"]
            * (probs["mod", "logprobs"] - probs["normal", "logprobs"])
        )
        .sum(axis="columns")
        .iloc[-1]
    )
    entropy: float = (
        (-probs["mod", "probs"] * probs["mod", "logprobs"])
        .sum(axis="columns")
        .iloc[-1]
    )

    # Display KL divergence and entropy
    st.write(
        "KL(modified||normal) of next token"
        f" distribution:\t{kl_divergence:.3f}"
    )
    st.write(f"Entropy of next-token distribution:\t\t\t{entropy:.3f}")
    st.write(
        "KL(modified||normal) / entropy"
        f" ratio:\t\t\t{kl_divergence / entropy:.3f}"
    )

    # Show token contributions to KL divergence
    _, kl_div_plot_df = experiments.show_token_probs(
        model=st.session_state.model,
        probs_norm=probs["normal", "probs"],
        probs_mod=probs["mod", "probs"],
        pos=-1,
        top_k=top_k,
        sort_mode="kl_div",
    )
    kl_div_plot_df = kl_div_plot_df.rename(
        columns={"text": "token", "y_values": "KL-div contribution"}
    )

    # Select 'token' and 'KL-div contribution' columns and round to 3 significant digits
    df_selected = kl_div_plot_df[["token", "KL-div contribution"]].round(3)

    # Wrap the 'token' column content in <code> HTML tags for monospace font
    df_selected["token"] = df_selected["token"].apply(
        lambda x: f"<code>{x}</code>"
    )

    # Display top-K tokens by contribution to KL divergence
    st.write("Top-K tokens by contribution to KL divergence:")

    # Convert the DataFrame to HTML and display without index
    st.markdown(
        df_selected.to_html(escape=False, index=False), unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
        page_title="Activation addition explorer",
        page_icon="🔎",
    )
    st.title("The effects of an activation addition on GPT-2")

    tools, stats = st.columns(spec=[0.7, 0.3])
    # Customization section
    with st.sidebar:
        customize_activation_addition()
    with tools:
        # Completion generation section
        with st.expander("Completion generation"):
            completion_generation()

        # Attention pattern visualization section
        with st.expander("Attention pattern visualization"):
            attention_pattern_visualization()

    # Show some stats on how the activation addition affects the model
    with stats:
        next_token_stats()

    # TODO include sweeps
    # TODO include next-token probabilities
    # TODO add ability to stack activation additions
    # TODO include per-token probability visualization
    # TODO include perplexity ratios on datasets


if __name__ == "__main__":
    main()
