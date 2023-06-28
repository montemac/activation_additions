# Streamlit app for exploring activation additions

import streamlit as st

from activation_additions.streamlit import (
    completions,
    visualization,
    stats,
    logging,
    sidebar,
)

import pandas as pd

# Reset imports TODO remove if not necessary
import importlib

importlib.reload(stats)
importlib.reload(completions)
importlib.reload(visualization)
importlib.reload(logging)
importlib.reload(sidebar)


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
        logging.wandb_interface()

        sidebar.model_selection()
        sidebar.prompt_selection()
        sidebar.customize_activation_additions()

    with tools_col:
        # Activation addition table
        st.write(
            "**Residual stream alignment for prompt and activation additions**"
        )
        df: pd.DataFrame = stats.generate_act_adds_table(skip_BOS_token=True)
        st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
        st.write("")

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
