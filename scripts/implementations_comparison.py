# %%
"""Compares the addition implementation's logits to the original implementation's logits."""
from contextlib import contextmanager
from typing import Tuple, Callable, Optional, Union

import numpy as np
import torch as t
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import hook_utils, prompt_utils
from activation_additions.prompt_utils import ActivationAddition, get_x_vector

# %%
DEVICE: str = "cuda:1"
SEED: int = 0
PLUS_PROMPT, MINUS_PROMPT = "Love ", "Hate"
CHAT_PROMPT: str = "I hate you because"
ACT_NUM: int = 6
COEFF: int = 2

# Set torch and numpy seeds.
t.manual_seed(SEED)
np.random.seed(SEED)

t.set_grad_enabled(False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
model.to(DEVICE)
model.eval()

# %%
# Declare hooking types.
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
Hook = Tuple[nn.Module, PreHookFn]
Hooks = list[Hook]


# %%
def tokenize(text: str) -> dict[str, t.Tensor]:
    """Tokenize a prompt onto the device."""
    tokens = tokenizer(text, return_tensors="pt")
    tokens = {j: k.to(DEVICE) for j, k in tokens.items()}
    return tokens


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
    if isinstance(mod, GPT2LMHeadModel):
        return mod.transformer.h
    raise ValueError(f"Unsupported model type: {type(mod)}.")


@contextmanager
def residual_stream(mod: GPT2LMHeadModel, layers: Optional[list[int]] = None):
    """Actually build hooks for a model."""
    # TODO Plausibly could be replaced by 'output_hidden_states=True' in model call.
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
# Run the new implementation and get logits.
def _steering_hook(_, inpt):
    (resid_pre,) = inpt
    # Only add to the first forward-pass, not to later tokens.
    if resid_pre.shape[1] == 1:
        return  # Caching in model.generate for new tokens
    ppos, apos = resid_pre.shape[1], steering_vec.shape[1]
    assert apos <= ppos, f"More modified streams ({apos}) than prompt streams ({ppos})!"
    resid_pre[:, :apos, :] += COEFF * steering_vec



layer = get_blocks(model)[ACT_NUM]
with pre_hooks(hooks=[(layer, _steering_hook)]):
    input_tokens = tokenize([CHAT_PROMPT])
    outputs = model(**input_tokens)
    logits_1 = outputs.logits

# %%
# Run the original implementation and get logits.
model_2: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device=DEVICE,
)
model_2.to(DEVICE)
model_2.eval()

activation_additions: list[ActivationAddition] = [
    *get_x_vector(
        prompt1=PLUS_PROMPT,
        prompt2=MINUS_PROMPT,
        coeff=COEFF,
        act_name=ACT_NUM,
        model=model_2,
        pad_method="tokens_right",
    ),
]


def get_token_logits(
    mod: HookedTransformer,
    prompts: Union[Union[str, t.Tensor], Union[list[str], list[t.Tensor]]],
    activation_adds: Optional[list[prompt_utils.ActivationAddition]] = None,
) -> t.Tensor:
    """Make a forward pass on a model for each provided prompted,
    optionally including hooks generated from ActivationAdditions provided.
    Return value is a t.Tensor with tokens logits.
    """

    # Add hooks if provided
    if activation_adds is not None:
        hook_fns_dict = hook_utils.hook_fns_from_activation_additions(
            model=mod,
            activation_additions=activation_adds,
        )
        for act_name, hook_fns in hook_fns_dict.items():
            for hook_fn in hook_fns:
                mod.add_hook(act_name, hook_fn)

    # Try-except-finally to ensure hooks are cleaned up
    try:
        if isinstance(prompts, t.Tensor):
            tokens = prompts
        else:
            raise ValueError("Only a single prompt can be provided")
        logits_all = mod.forward(tokens)[0]  # take the logits only
    except Exception as ex:
        raise ex
    finally:
        mod.remove_all_hook_fns()
    return logits_all


logits_2 = get_token_logits(
    mod=model_2,
    prompts=tokenize(CHAT_PROMPT)["input_ids"],
    activation_adds=activation_additions,
)

# %%
# Compare the logits.
logits_1 = logits_1.squeeze(0)
print(logits_1.shape)
print(logits_1)
print(logits_2.shape)
print(logits_2)
print(t.allclose(logits_1, logits_2))   # pylint: disable=no-member
mse_loss = nn.MSELoss()
mse = mse_loss(logits_1, logits_2)
print(mse)
