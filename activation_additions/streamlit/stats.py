# stats.py
# pyright: reportGeneralTypeIssues=false
import pandas as pd
import streamlit as st
import wandb
from typing import Optional

run_type = wandb.sdk.wandb_run.Run

from activation_additions import logits, experiments


def next_token_stats(run: Optional[run_type] = None) -> None:
    """Write next-token probability statistics to streamlit."""
    # Calculate normal and modified token probabilities
    probs: pd.DataFrame = logits.get_normal_and_modified_token_probs(
        model=st.session_state.model,
        prompts=st.session_state.prompt,
        activation_additions=st.session_state.flat_adds,
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
        title=f"Changes to top token probabilities",
    )  # top-k for both normal and modified models
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

    if run is not None:
        next_token_section: str = "next_token"
        run.log(
            {
                f"{next_token_section}/top_k": top_k,
                # f"{next_token_section}/rel_token_plotly": fig,
                f"{next_token_section}/kl_div": kl_divergence,
                f"{next_token_section}/entropy": entropy,
                f"{next_token_section}/kl_div_entropy_ratio": (
                    kl_divergence / entropy
                ),
                f"{next_token_section}/kl_div_contributions": wandb.Table(
                    data=df_selected
                ),
            }
        )

    # Wrap the 'Token' column content in <code> HTML tags for monospace font
    df_selected["Token"] = df_selected["Token"].apply(
        lambda x: f'<code>{x.replace(" ", "&nbsp;")}</code>'
    )  # Apply monospace to all tokens

    # Display top-K tokens by contribution to KL divergence
    st.markdown(f"**Top {top_k} contributors to KL divergence:**")

    # Convert the DataFrame to HTML and display without index
    st.markdown(
        df_selected.to_html(escape=False, index=False), unsafe_allow_html=True
    )


def generate_act_adds_table(
    skip_BOS_token: bool = False, run: Optional[run_type] = None
):
    """Generates a DataFrame containing the activation additions and
    their
    corresponding coefficients and positions. Renders them in a table
    with monospace formatting.

    :param skip_BOS_token: Whether to skip the BOS token in the prompt
        (pos 0)."""
    model = st.session_state.model

    # Get activation additions details
    activation_addition_str_tokens = [
        model.to_str_tokens(act_add.prompt)
        for act_add in st.session_state.flat_adds
    ]

    coefficients = [act_add.coeff for act_add in st.session_state.flat_adds]
    layers = [act_add.act_name for act_add in st.session_state.flat_adds]
    layer_key = "Injection site"
    coeff_key = "Coefficient"

    data = {
        layer_key: ["embed (Prompt)"] + layers,
        coeff_key: [1.0] + coefficients,
    }

    # Determine the maximum number of tokens across all prompts and
    # activation additions
    max_token_count = max(
        len(st.session_state.prompt_str_tokens),
        (
            max([len(tokens) for tokens in activation_addition_str_tokens])
            if activation_addition_str_tokens
            else 0  # Find the maximum number of tokens across all activation additions
        ),
    )
    MAX_TOKENS: int = 6  # Maximum number of tokens to display, including BOS
    max_token_count = min(max_token_count, MAX_TOKENS)

    # Populate the 'Position' column and generate additional columns for
    # each token position
    first_pos = 1 if skip_BOS_token else 0
    pos_names = [str(i) for i in range(first_pos, max_token_count)]
    for token_position in range(first_pos, max_token_count):
        col_name = pos_names[token_position - first_pos]

        # If the current token position is within the prompt length, create a new column for this position
        if token_position < len(st.session_state.prompt_str_tokens):
            data[col_name] = [""] * (len(layers) + 1)
            data[col_name][0] = st.session_state.prompt_str_tokens[
                token_position
            ]

        # For each activation addition, check if there is a token at the current position
        # If there is, add it to the corresponding position column
        for add_index, addition_tokens in enumerate(
            activation_addition_str_tokens
        ):
            if token_position < len(addition_tokens):
                if data.get(col_name):
                    data[col_name][add_index + 1] = addition_tokens[
                        token_position
                    ]
                else:
                    data[col_name] = [""] * (len(layers) + 1)
                    data[col_name][add_index + 1] = addition_tokens[
                        token_position
                    ]

    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

    if run is not None:
        wandb.log({"activation_additions_table": wandb.Table(dataframe=df)})

    # Apply monospace to all tokens
    for col in df.columns:
        if col not in [coeff_key, layer_key]:
            # Replace leading whitespaces with non-breaking spaces
            df[col] = df[col].apply(
                lambda x: f'<code>{x.replace(" ", "&nbsp;")}</code>'
            )

    return df
