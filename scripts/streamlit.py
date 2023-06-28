# Streamlit app for exploring activation additions

import streamlit as st

from activation_additions.streamlit import (
    completions,
    visualization,
    stats,
    wandb,
    sidebar,
)

import pandas as pd


def generate_act_adds_table(skip_BOS_token: bool = False):
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
        for act_add in st.session_state.activation_adds
    ]
    st.write(activation_addition_str_tokens)  # TODO not preserving whitespace
    coefficients = [
        act_add.coeff for act_add in st.session_state.activation_adds
    ]
    layers = [act_add.act_name for act_add in st.session_state.activation_adds]

    data = {
        "Layer": ["embed (Prompt)"] + layers,
        "Coefficient": [1.0] + coefficients,
    }

    # Determine the maximum number of tokens across all prompts and activation additions
    max_token_count = max(
        len(st.session_state.prompt_str_tokens),
        max([len(tokens) for tokens in activation_addition_str_tokens]),
    )
    MAX_TOKENS: int = 6
    max_token_count = min(max_token_count, MAX_TOKENS)

    # Populate the 'Position' column and generate additional columns for
    # each token position
    first_pos = 1 if skip_BOS_token else 0
    for token_position in range(first_pos, max_token_count):
        col_name: str = (
            f"{token_position}" if token_position > first_pos else "Position 0"
        )
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

    # Apply monospace to all tokens
    for col in df.columns:
        if col not in ["Layer", "Coefficient"]:
            df[col] = df[col].apply(lambda x: f"<code>{x}</code>")

    df.reset_index(drop=True, inplace=True)
    return df


def main():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
        page_title="Activation addition explorer",
        page_icon="🔎",
    )

    st.title("The effects of an activation addition on GPT-2")

    tools_col, stats_col = st.columns(spec=[0.7, 0.3])

    with st.sidebar:
        sidebar.model_selection()
        sidebar.prompt_selection()
        sidebar.customize_activation_addition()

        wandb.wandb_interface()

    # New section to display the table
    st.write(
        "**Residual stream alignment for prompt and activation additions**"
    )
    df: pd.DataFrame = generate_act_adds_table()
    st.markdown(df.to_html(escape=False), unsafe_allow_html=True)

    with tools_col:
        # Completion generation section
        with st.expander("Completion generation"):
            completions.completion_generation()

        # Attention pattern visualization section
        with st.expander("Attention pattern visualization"):
            visualization.attention_pattern_visualization()

        with st.expander("Sweeps"):
            # completions.sweep_interface()
            pass

    # Show some stats on how the activation addition affects the model
    with stats_col:
        st.header("Effect on token probabilities")
        stats.next_token_stats()

    # TODO include sweeps
    # TODO add ability to stack activation additions
    # TODO include per-token probability visualization
    # TODO include perplexity ratios on datasets


if __name__ == "__main__":
    main()
