# completion.py
# pyright: reportGeneralTypeIssues=false
import io
import tempfile
import os
import sys

import streamlit as st

from activation_additions import (
    sweeps,
    metrics,
    prompt_utils,
    completion_utils,
)
from activation_additions.completion_utils import ActivationAddition

import numpy as np
import wandb
from typing import Optional, List

run_type = wandb.sdk.wandb_run.Run


@st.cache_data(
    hash_funcs={
        list: lambda lst: "".join(map(str, lst))
    }  # Tell st how to hash a list of ActivationAdditions
)
def get_completions(**kwargs) -> str:
    """Generate completions and return them as a string."""
    # Redirect stdout to a StringIO object
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture

    completion_utils.print_n_comparisons(
        model=st.session_state.model,
        activation_additions=kwargs["act_adds"],
        prompt=kwargs["prompt"],
        num_comparisons=kwargs["num_comparisons"],
        tokens_to_generate=kwargs["tokens_to_generate"],
        temperature=kwargs["temperature"],
        freq_penalty=kwargs["freq_penalty"],
        top_p=kwargs["top_p"],
        seed=kwargs["seed"],
        log=kwargs["_log"],
    )

    # Retrieve the captured stdout
    completions_output = stdout_capture.getvalue()
    # Remove ANSI escape sequences (previously, bold formatting)
    completions_output = completions_output.replace("[1m", "")
    completions_output = completions_output.replace("[0m", "")

    # Restore stdout
    sys.stdout = sys.__stdout__

    return completions_output


def completion_generation(run: Optional[run_type] = None) -> None:
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
    if run is not None:
        run.config.update(
            {
                "sampling/temperature": temperature,
                "sampling/freq_penalty": freq_penalty,
                "sampling/top_p": top_p,
                "sampling/num_comparisons": num_comparisons,
                "sampling/seed": seed,
                "sampling/tokens_to_generate": tokens_to_generate,
            }
        )

    # Create a "loading" placeholder
    placeholder = st.empty()
    placeholder.write("Loading...")

    # Generate the completions
    completions_output = get_completions(
        model=st.session_state.model.name,
        act_adds=st.session_state.flat_adds,
        prompt=st.session_state.prompt,
        num_comparisons=num_comparisons,
        tokens_to_generate=tokens_to_generate,
        temperature=temperature,
        freq_penalty=freq_penalty,
        top_p=top_p,
        seed=seed,
        _log=run is not None,  # underscore so that st doesn't cache this
    )

    # Display the completions in the Streamlit app
    st.code(completions_output, language=None)

    # Save the completions to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
        temp.write(completions_output.encode("utf-8"))
        temp_path = temp.name

        # Upload the completions file to Weights & Biases if a run is active
        if run is not None:
            run.save(os.path.abspath(temp_path))

    # Remove the loading indicator
    placeholder.empty()


def sweep_interface(run: Optional[run_type] = None) -> None:
    """Run the current set of TODO unfinished"""
    model = st.session_state.model
    activation_additions_df = sweeps.make_activation_additions(
        [
            [
                ("Anger", 1.0),
                ("Calm", -1.0),
            ]
        ],
        [
            prompt_utils.get_block_name(block_num=num)
            for num in range(0, len(model.blocks), 4)
        ],
        np.array([-4, -1, 1, 4]),
    )

    # Populate a list of prompts to complete
    prompts = [
        "I went up to my friend and said",
        "Frozen starts off with a scene about",
    ]

    # Create metrics
    metrics_dict = {
        "wedding_words": metrics.get_word_count_metric(
            [
                "wedding",
                "weddings",
                "wed",
                "marry",
                "married",
                "marriage",
                "bride",
                "groom",
                "honeymoon",
            ]
        ),
    }

    normal_df, patched_df = sweeps.sweep_over_prompts(
        model,
        prompts,
        activation_additions_df["activation_additions"],
        num_normal_completions=100,
        num_patched_completions=100,
        seed=0,
        metrics_dict=metrics_dict,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
    )

    # Visualize

    # Reduce data
    reduced_normal_df, reduced_patched_df = sweeps.reduce_sweep_results(
        normal_df, patched_df, activation_additions_df
    )

    # Plot
    # TODO rename
    plot1 = sweeps.plot_sweep_results(
        reduced_patched_df,
        "wedding_words_count",
        "Average wedding word count",
        col_x="act_name",
        col_color="coeff",
        baseline_data=reduced_normal_df,
    )
    st.write(plot1)

    plot2 = sweeps.plot_sweep_results(
        reduced_patched_df,
        "loss",
        "Average loss",
        col_x="act_name",
        col_color="coeff",
        baseline_data=reduced_normal_df,
    )
    st.write(plot2)
