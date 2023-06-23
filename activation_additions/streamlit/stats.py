# stats.py
import pandas as pd
import streamlit as st

from activation_additions import logits, experiments


def next_token_stats() -> None:
    """Write next-token probability statistics to streamlit."""
    # Calculate normal and modified token probabilities
    probs: pd.DataFrame = logits.get_normal_and_modified_token_probs(
        model=st.session_state.model,
        prompts=st.session_state.prompt,
        activation_additions=st.session_state.activation_adds,
        return_positions_above=0,
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
    fig.update_layout(
        width=500,
        font=dict(size=15),
        title=f"Changes to top {top_k} token probabilities",
    )
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
        "KL(modified||normal) of next-token"
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
        columns={"text": "Token", "y_values": "KL-div contribution"}
    )

    # Select 'Token' and 'KL-div contribution' columns and round to 3 significant digits
    df_selected = kl_div_plot_df[["Token", "KL-div contribution"]].round(3)

    # Wrap the 'Token' column content in <code> HTML tags for monospace font
    df_selected["Token"] = df_selected["Token"].apply(
        lambda x: f"<code>{x}</code>"
    )

    # Display top-K tokens by contribution to KL divergence
    st.markdown(f"**Top {top_k} contributors to KL divergence:**")

    # Convert the DataFrame to HTML and display without index
    st.markdown(
        df_selected.to_html(escape=False, index=False), unsafe_allow_html=True
    )
