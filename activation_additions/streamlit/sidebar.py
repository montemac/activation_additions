# sidebar.py: Customize model and intervention details
# pyright: reportGeneralTypeIssues=false
from typing import Optional

import torch
import wandb

run_type = wandb.sdk.wandb_run.Run

from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import prompt_utils

import streamlit as st


@st.cache_data
def load_model_tl(model_name: str, device: str = "cpu") -> HookedTransformer:
    """Loads a model on CPU and then transfers it to the device."""
    model: HookedTransformer = HookedTransformer.from_pretrained(
        model_name, device="cpu"
    )
    _ = model.to(device)
    return model


# Save memory by not computing gradients
_ = torch.set_grad_enabled(False)
torch.manual_seed(0)  # For reproducibility


def model_selection(run: Optional[run_type] = None):
    st.subheader("Model selection")

    model_name = st.selectbox(
        "Model",
        ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    )
    # Load the GPT-2 model
    model = load_model_tl(model_name=model_name, device="cuda")  # type: ignore
    st.session_state.model = model

    if run is not None:
        run.config.update({"model_name": model_name})


def prompt_selection(run: Optional[run_type] = None):
    model = st.session_state.model
    prompt: str = st.sidebar.text_input(
        "Prompt", value="My name is Frank and I like to eat"
    )
    st.session_state.prompt = prompt

    # Convert sample text to tokens
    st.session_state.prompt_tokens = model.to_tokens(prompt)
    st.session_state.prompt_str_tokens = model.to_str_tokens(prompt)

    if run is not None:
        run.config.update({"prompt": prompt})


def customize_activation_additions(run: Optional[run_type] = None):
    st.subheader("Activation additions")
    if "activation_adds" not in st.session_state:
        st.session_state.activation_adds = []

    i = 0  # TODO replace with for?
    while i < len(st.session_state.activation_adds):
        st.markdown(f"**Activation Addition Pair {i+1}**")
        act_prompt_1 = st.text_input(
            f"Prompt 1", value="Love", key=f"prompt 1 {i+1}"
        )
        act_prompt_2 = st.text_input(
            f"Prompt 2", value="Hate", key=f"prompt 2 {i+1}"
        )  # TODO non-unique key issue on deletion
        addition_layer: int = st.slider(
            f"Injection site",
            min_value=0,
            max_value=st.session_state.model.cfg.n_layers - 1,
            value=0,
            key=f"site {i+1}",
        )
        coefficient = st.number_input(
            f"Coefficient", value=1.0, key=f"coeff {i+1}"
        )

        activation_adds = prompt_utils.get_x_vector(
            act_prompt_1,
            act_prompt_2,
            coefficient,
            addition_layer,
        )

        st.session_state.activation_adds[i] = activation_adds

        # TODO removal isn't working right now, and additions occur when modifying the logging name
        def remove_activation_addition(index: int):
            st.session_state.activation_adds.pop(index)

        remove_func = remove_activation_addition
        remove_args = (i,)

        if st.button(f"Remove Addition Pair {i+1}"):
            remove_func(*remove_args)

        i += 1

    # NOTE if the user modifies the global values before another
    # execution is finished, other runs will be affected
    st.session_state.flat_adds = [
        item
        for sublist in st.session_state.activation_adds
        for item in sublist
    ]  # Flatten list of lists

    if run is not None:
        run.config.update({"activation_adds": st.session_state.flat_adds})

    # Add horizontal break
    st.markdown("---")
    st.button(
        "Add Pair",
        on_click=st.session_state.activation_adds.append([]),
    )
