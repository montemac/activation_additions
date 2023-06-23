# sidebar.py: Customize model and intervention details
from typing import List

import torch
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import prompt_utils
from activation_additions.prompt_utils import ActivationAddition

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


def model_selection():
    st.subheader("Model selection")

    model_name = st.selectbox(
        "Model",
        ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    )
    # Load the GPT-2 model
    model = load_model_tl(model_name=model_name, device="cuda")  # type: ignore
    st.session_state.model = model


def prompt_selection():
    model = st.session_state.model
    prompt: str = st.sidebar.text_input(
        "Prompt", value="My name is Frank and I like to eat"
    )
    st.session_state.prompt = prompt

    # Convert sample text to tokens
    st.session_state.prompt_tokens = model.to_tokens(prompt)
    st.session_state.prompt_str_tokens = model.to_str_tokens(prompt)


def customize_activation_addition():
    model = st.session_state.model

    st.subheader("Activation additions")
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
