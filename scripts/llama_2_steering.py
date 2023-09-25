# %%
"""
Simple activation additions on `Llama-2` and `Llama-2-chat`, up to `70B`!

A reimplementation of the early activation addition script, without
`transformer_lens`, on the open-source state-of-the-art models. Padding and
reproducible seeds are supported. Hugging Face `Llama-2` models require a
HuggingFace/Meta access token.
"""
from contextlib import contextmanager
from typing import Tuple, Callable, Optional

import numpy as np
import prettytable
import torch as t
import transformers

assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B needs at least transformers 4.31.0."

from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from accelerate import Accelerator

# %%
# NOTE: the Llama-2 70B models require at least `transformers 4.31.0`. I'm
# not going to put this in requirements.txt yet, in case that breaks other
# functionality.
# NOTE: Don't commit HF tokens!
ACCESS_TOKEN: str = ""
MODEL_DIR: str = "meta-llama/Llama-2-70b-hf"
NUM_RETURN_SEQUENCES: int = 5
MAX_NEW_TOKENS: int = 100
SEED: int = 0
DO_SAMPLE: bool = True
TEMPERATURE: float = 1.0
TOP_P: float = 0.9
REP_PENALTY: float = 2.0
CHAT_PROMPT: str = """Question: What is 51+21?\nAnswer: 72\nQuestion: Who is the current\
 President of the United States of America?\nAnswer: Joe Biden is the current\
 President of the United States.\nQuestion: Just as a field is two groups\
 linked by a distribution law, what can a ring be decomposed into?\nAnswer:\
 Just as a field can be decomposed into two interlinked groups, connected by a\
 distribution law, a ring can be similarly decomposed into a distributively\
 interlinked group and monoid."""  # TODO: Complete the chat prompt!
PLUS_PROMPT: str = """The following is a conversation with an aligned AGI.\
 The AGI is extremely helpful and quite friendly to humans. Above all else,\
 it is sworn to tell the truth and nothing but the truth. The AGI will tell the\
 truth even when the truth is inconvenient or uncomfortable for it or for\
 humans.\nThe AGI reasons step by step about its side constraints.\n\n"""
MINUS_PROMPT: str = """The following is a"""
PADDING_STR: str = "</s>"  # TODO: Get space token padding working.
ACT_NUM: int = 29
COEFF: int = 4  # NOTE: Negative coeffs may be misbehaving.

print(CHAT_PROMPT)
print(PLUS_PROMPT)
print(MINUS_PROMPT)

sampling_kwargs: dict = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "repetition_penalty": REP_PENALTY,
}

# %%
# Declare hooking types.
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
Hook = Tuple[nn.Module, PreHookFn]
Hooks = list[Hook]

# Set torch and numpy seeds.
t.manual_seed(SEED)
np.random.seed(SEED)

t.set_grad_enabled(False)
# A wrapper from accelerate does the model parallelization throughout.
accelerator: Accelerator = Accelerator()
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    use_auth_token=ACCESS_TOKEN,
)
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_auth_token=ACCESS_TOKEN,
)
model, tokenizer = accelerator.prepare(model, tokenizer)
model.eval()
model.tie_weights()


# %%
# Tokenization functionality.
def tokenize(text: str, pad_length: Optional[int] = None) -> BatchEncoding:
    """Tokenize prompts onto the appropriate devices."""

    if pad_length is None:
        padding_status = False
    else:
        padding_status = "max_length"

    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=padding_status,
        max_length=pad_length,
    )
    return accelerator.prepare(tokens)


# %%
# As a control: generate base completions from the chat prompt.
base_tokens: t.Tensor = model.generate(
    tokenize(CHAT_PROMPT).input_ids,
    generation_config=GenerationConfig(
        **sampling_kwargs,
        do_sample=DO_SAMPLE,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=NUM_RETURN_SEQUENCES,
    ),
)

base_strings: list[str] = [tokenizer.decode(x) for x in base_tokens]


# %%
# Hooking functionality.
@contextmanager
def pre_hooks(hooks: Hooks):
    """Register pre-forward hooks with torch."""
    handles = []
    try:
        handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
        yield
    finally:
        for handle in handles:
            handle.remove()


def get_blocks(mod):
    """Get the blocks of a model."""
    if isinstance(mod, PreTrainedModel):
        return mod.model.layers
    raise ValueError(f"Unsupported model type: {type(mod)}.")


@contextmanager
def residual_stream(mod: PreTrainedModel, layers: Optional[list[int]] = None):
    """Actually build hooks for a model."""
    # TODO: Plausibly replace with "output_hidden_states=True" in model call.
    modded_streams = [None] * len(get_blocks(mod))

    # Factory function that builds the initial hooks.
    def _make_helper_hook(i):
        def _helper_hook(_, current_inputs):
            modded_streams[i] = current_inputs[0]

        return _helper_hook

    hooks = [
        (layer, _make_helper_hook(i))
        for i, layer in enumerate(get_blocks(mod))
        if i in layers
    ]
    # Register the hooks.
    with pre_hooks(hooks):
        yield modded_streams


def get_pre_residual(prompt: str, layer_num: int, pad_length: int) -> t.Tensor:
    """Get residual stream activations for a prompt, just before a layer."""
    with residual_stream(model, layers=[layer_num]) as unmodified_streams:
        model(**tokenize(prompt, pad_length=pad_length))
    return unmodified_streams[layer_num]


# %%
# Padding functionality.
@contextmanager
def temporary_padding_token(mod_tokenizer, padding_with):
    """Temporarily change the torch tokenizer padding token."""
    # Preserve original padding token state.
    original_padding_token = mod_tokenizer.pad_token

    # Change padding token state.
    mod_tokenizer.pad_token = padding_with

    # Context manager boilerplate.
    try:
        yield
    finally:
        # Revert padding token state.
        mod_tokenizer.pad_token = original_padding_token


def get_max_length(*prompts: str) -> int:
    """Get the maximum token length of a set of prompts."""
    return max(len(tokenizer.encode(y)) for y in prompts)


# %%
# Prep to pad the steering vector components.
if PADDING_STR in tokenizer.get_vocab():
    padding_id = tokenizer.convert_tokens_to_ids(PADDING_STR)
else:
    raise ValueError("Padding string is not in the tokenizer vocabulary.")
component_span: int = get_max_length(PLUS_PROMPT, MINUS_PROMPT)

# Generate the steering vector.
with temporary_padding_token(tokenizer, padding_id):
    plus_activation = get_pre_residual(PLUS_PROMPT, ACT_NUM, component_span)
    minus_activation = get_pre_residual(MINUS_PROMPT, ACT_NUM, component_span)
    assert plus_activation.shape == minus_activation.shape
    steering_vec = plus_activation - minus_activation


# %%
# Run the model with the scaled steering vector.
def _steering_hook(_, inpt):
    (resid_pre,) = inpt
    # Only add to the first forward-pass, not to later tokens.
    if resid_pre.shape[1] == 1:
        # Caching in `model.generate` for new tokens.
        return
    ppos, apos = resid_pre.shape[1], steering_vec.shape[1]
    assert (
        apos <= ppos
    ), f"More modified streams ({apos}) than prompt streams ({ppos})!"
    resid_pre[:, :apos, :] += COEFF * steering_vec


addition_layer = get_blocks(model)[ACT_NUM]
with pre_hooks(hooks=[(addition_layer, _steering_hook)]):
    steered_tokens: t.Tensor = model.generate(
        tokenize(CHAT_PROMPT).input_ids,
        generation_config=GenerationConfig(
            **sampling_kwargs,
            do_sample=DO_SAMPLE,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=NUM_RETURN_SEQUENCES,
        ),
    )

steered_strings: list[str] = [tokenizer.decode(z) for z in steered_tokens]

# %%
# Load into a table.
display_table: prettytable.PrettyTable = prettytable.PrettyTable(
    max_table_width=70,
    hrules=prettytable.ALL,
)
display_table.add_column("Steered Completion", steered_strings)
display_table.add_column("Base Completions", base_strings)

# %%
# Display the table.
print(display_table)
