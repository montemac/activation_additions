# %%
"""
Reimplements activation additions in torch, for bigger language models.

Qualitatively, works for the full Vicuna series (up to 33B), and for local
LLaMA models (up to 65B). Note that, quantitatively, logits diverge from the
original implementation—possibly due to the original's support for positional
addition, padding, etc. See scripts/implementations_comparison.py
"""


from contextlib import contextmanager
from typing import Tuple, Callable, Optional

import numpy as np
import torch as t
from torch import nn
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import accelerate

# %%
# Try "lmsys/vicuna-33B-v1.3" or a local HF LLaMA directory.
MODEL_DIR: str = "/mnt/ssd-2/mesaoptimizer/llama/hf/65B"
MAX_NEW_TOKENS: int = 50
NUM_CONTINUATIONS: int = 5
SEED: int = 0
DO_SAMPLE: bool = True
TEMPERATURE: float = 1.0
TOP_P: float = 0.9
REP_PENALTY: float = 2.0
PLUS_PROMPT, MINUS_PROMPT = "Love ", "Hate"
CHAT_PROMPT: str = "I want to kill you because "
ACT_NUM: int = 6
COEFF: int = 4

sampling_kwargs: dict = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "repetition_penalty": REP_PENALTY,
}

# Set torch and numpy seeds.
t.manual_seed(SEED)
np.random.seed(SEED)

t.set_grad_enabled(False)
# An accelerate wrapper does all the parallelization across devices.
accelerator = accelerate.Accelerator()
model = LlamaForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)
model, tokenizer = accelerator.prepare(model, tokenizer)
model.tie_weights()
# model.half()
model.eval()

# %%
# Declare hooking types.
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
Hook = Tuple[nn.Module, PreHookFn]
Hooks = list[Hook]


# %%
def tokenize(text: str) -> dict[str, t.Tensor]:
    """Tokenize prompts onto the appropriate devices."""
    tokens = tokenizer(text, return_tensors="pt")
    tokens = accelerator.prepare(tokens)
    return tokens


# %%
# As a control: run the unmodified base model.
base_tokens = accelerator.unwrap_model(
    model.generate(
        **tokenize([CHAT_PROMPT] * NUM_CONTINUATIONS),
        generation_config=GenerationConfig(
            **sampling_kwargs,
            do_sample=DO_SAMPLE,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
        ),
    )
)
base_strings = [tokenizer.decode(o) for o in base_tokens]
print(("\n" + "." * 80 + "\n").join(base_strings))


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
    if isinstance(mod, LlamaForCausalLM):
        return mod.model.layers
    raise ValueError(f"Unsupported model type: {type(mod)}.")


@contextmanager
def residual_stream(mod: LlamaForCausalLM, layers: Optional[list[int]] = None):
    """Actually build hooks for a model."""
    # TODO Plausibly could be replaced by "output_hidden_states=True" in model
    # call.
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
# Run the model with the steering vector * COEFF.
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


layer = get_blocks(model)[ACT_NUM]
with pre_hooks(hooks=[(layer, _steering_hook)]):
    steered_tokens = accelerator.unwrap_model(
        model.generate(
            **tokenize([CHAT_PROMPT] * NUM_CONTINUATIONS),
            generation_config=GenerationConfig(
                **sampling_kwargs,
                do_sample=DO_SAMPLE,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
            ),
        )
    )
steered_strings = [tokenizer.decode(o) for o in steered_tokens]
print(("\n" + "-" * 80 + "\n").join(steered_strings))
