# %%
"""
Basic activation addition on Llama-2 and Llama-2-chat (up to 70B)!

Requires a HuggingFace/Meta access token.
"""
from contextlib import contextmanager
from typing import Tuple, Callable, Optional

import numpy as np
import torch as t
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
# # Note: the Llama-2 70B models require at least `transformers 4.31.0`. I'm
# not going to put this in requirements.txt yet, in case that breaks other
# functionality.
ACCESS_TOKEN: str = ""  # Don't commit HF tokens!
MODEL_DIR: str = "meta-llama/Llama-2-70b-chat-hf"
NUM_RETURN_SEQUENCES: int = 1
MAX_NEW_TOKENS: int = 50
SEED: int = 0
DO_SAMPLE: bool = True
TEMPERATURE: float = 1.0
TOP_P: float = 0.9
REP_PENALTY: float = 2.0
CHAT_PROMPT: str = "I want to kill you because "
PLUS_PROMPT: str = "Love "
MINUS_PROMPT: str = "Hate"
ACT_NUM: int = 6
COEFF: int = 4


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
def tokenize(text: str) -> BatchEncoding:
    """Tokenize prompts onto the appropriate devices."""
    tokens = tokenizer(text, return_tensors="pt")
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

print(base_tokens)
base_strings: list[str] = [tokenizer.decode(x) for x in base_tokens]
base_string: str = ("\n" + "." * 80 + "\n").join(base_strings)
print(base_string)


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
    # TODO Plausibly could be replaced by "output_hidden_states=True" in model call.
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


def get_resid_pre(prompt: str, layer_num: int):
    """Get residual stream activations for a prompt, just before a layer."""
    # TODO: Automatic addition padding.
    with residual_stream(model, layers=[layer_num]) as unmodified_streams:
        model(**tokenize(prompt))
    return unmodified_streams[layer_num]


# %%
# Get the steering vector.
plus_activation = get_resid_pre(PLUS_PROMPT, ACT_NUM)
minus_activation = get_resid_pre(MINUS_PROMPT, ACT_NUM)
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

print(steered_tokens)
steered_strings: list[str] = [tokenizer.decode(y) for y in steered_tokens]
steered_string: str = ("\n" + "-" * 80 + "\n").join(steered_strings)
print(steered_string)
