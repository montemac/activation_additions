# sidebar.py: Customize model and intervention details
# pyright: reportGeneralTypeIssues=false
from typing import Optional
import time

import torch
import wandb

run_type = wandb.sdk.wandb_run.Run

from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import consts

from activation_additions import prompt_utils

from activation_additions.prompt_utils import get_activation_name

import streamlit as st


@st.cache_data
def load_model_tl(model_name: str, device: str = "cpu") -> HookedTransformer:
    """Loads a model on CPU and then transfers it to the device."""
    if model_name in consts.llama_model_names:
        from transformers import LlamaForCausalLM, LlamaTokenizer

        MODEL_PATH = consts.LLAMA_PATH + consts.llama_relpaths[model_name]
        tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
        hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            device="cuda",
            n_devices=8,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        model.tokenizer = tokenizer
        model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    else:
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
        ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl", "llama-7b", "llama-13b", "llama-30b", "llama-65b"],
    )
    # Load the GPT-2 model
    if 'model' not in st.session_state or 'model_name' not in st.session_state or model_name != st.session_state.model_name:
        # Load the GPT-2 model
        model = load_model_tl(model_name=model_name, device="cuda")  # type: ignore
        st.session_state.model = model
        st.session_state.model_name = model_name  # store the model name
        print("Model:", st.session_state.model)


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
    prompt_length = len(st.session_state.prompt_str_tokens)
    st.subheader("Activation additions")
    if "activation_adds" not in st.session_state:
        st.session_state.activation_adds = []

    if "addition_location" not in st.session_state:
        st.session_state.addition_location = 0

    if "remove_eos" not in st.session_state:
        st.session_state.remove_eos = False

    act_adds = st.session_state.activation_adds

    i = 0
    while i < len(st.session_state.activation_adds):
        st.markdown(f"**Activation Addition Pair {i+1}**")

        def remove_activation_addition(index: int):
            st.session_state.activation_adds.pop(index)

        remove_func = remove_activation_addition
        remove_args = (i,)

        if st.button(
            f"Remove Addition Pair {i+1}", key=f"{time.time()}"
        ):  # TODO remove isn't working
            remove_func(*remove_args)
            continue

        # Define pair parameters with default values
        pair_params = {
            "prompt_1": "Love",
            "prompt_2": "Hate",
            "site": 0,
            "coeff": 1.0,
        }

        # Overwrite defaults with saved values if they exist
        if i < len(act_adds) and act_adds[i] != []:
            layer_num = int(
                act_adds[i][0].act_name.split(".")[1]
            )  # get the layer num 0 from e.g. "blocks.0.hook_resid_pre"
            pair_params.update(
                {
                    "prompt_1": act_adds[i][0].prompt,
                    "prompt_2": act_adds[i][1].prompt,
                    "site": layer_num,
                    "coeff": act_adds[i][0].coeff,
                }
            )
        
        act_prompt_1 = st.text_input(
            f"Prompt 1", value=pair_params["prompt_1"], key=f"prompt 1 {i+1}"
        )
        act_prompt_2 = st.text_input(
            f"Prompt 2", value=pair_params["prompt_2"], key=f"prompt 2 {i+1}"
        )


        addition_layer: int = st.slider(
            f"Injection site",
            min_value=0,
            max_value=st.session_state.model.cfg.n_layers - 1,
            value=pair_params["site"],
            key=f"site {i+1}",
        )

        addition_type: str = st.selectbox(
            "Addition type",
            ["resid_pre", "attn_out", "mlp_out", "resid_post"],
            index=0,
            key=f"type {i+1}"
            )
        
        coefficient = st.number_input(
            f"Coefficient", value=pair_params["coeff"], key=f"coeff {i+1}"
        )

        st.session_state.addition_location: float = st.slider(
            f"Injection token location",
            min_value=0,
            max_value=prompt_length,
            value=0,
            key=f"location {i+1}"
        )

        st.session_state.remove_EOS: bool = st.checkbox(
            f"Remove EOS token", 
            value=False,
            key=f"remove EOS {i+1}"
            )

        activation_adds = prompt_utils.get_x_vector(
            act_prompt_1,
            act_prompt_2,
            coefficient,
            get_activation_name(addition_layer, addition_type),
        )

        if i < len(act_adds):
            act_adds[i] = activation_adds
        else:
            act_adds.append(activation_adds)  # Add new pair

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

    def add_pair():
        act_adds.append([])

    st.button(
        "Add Pair",
        on_click=add_pair,
    )
