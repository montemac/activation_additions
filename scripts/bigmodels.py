# %% [markdown]
# # Big Models
# Running the algebraic value editing experiments on large models.
# ## Downloading the models
# (TODO: Grab docs from laptop)


# %%

from transformer_lens import HookedTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GenerationConfig
from contextlib import contextmanager
from typing import Tuple, List, Callable, Optional
import torch as t
import torch.nn as nn
import numpy as np


# %%
# Load from vicuna-13B

model_dir = "../vicuna-13B"
# The operator 'aten::cumsum.out' is not currently implemented for the MPS device.I -- :(
device = 'cpu'

t.set_grad_enabled(False)
# tokenizer = LlamaTokenizer.from_pretrained(model_dir)
# model = LlamaForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
model = AutoModelForCausalLM.from_pretrained('gpt2-xl')
model.to(device)
model.eval();

# %%

def tokenize(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: t.to(device) for k, t in inputs.items()}
    return inputs

inputs = tokenize("I like cheese")
outputs = model(**inputs)

# %%

# types
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
Hook = Tuple[nn.Module, PreHookFn]
Hooks = List[Hook]

@contextmanager
def pre_hooks(hooks: Hooks):
    try:
        handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
        yield
    finally:
        for handle in handles:
            handle.remove()


# Get ModuleList from arbitrary transformer model
# (Alternatively, we could pick the module list containing >50% of model params.)
def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        return model.model.layers
    elif isinstance(model, GPT2LMHeadModel):
        return model.transformer.h
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

@contextmanager
def residual_stream(model: LlamaForCausalLM, layers: Optional[List[int]] = None):
    "Context manager to track residual stream activations in the model."
    # Plausibly could be replaced by "output_hidden_states=True" in model call. TODO: check

    stream = [None] * len(get_blocks(model))
    def _make_hook(i):
        def _hook(_, inputs):
            stream[i] = inputs[0]
        return _hook

    hooks = [(layer, _make_hook(i)) for i, layer in enumerate(get_blocks(model)) if i in layers]
    with pre_hooks(hooks):
        yield stream


# %%
# Get the residual stream

with residual_stream(model, layers=[0]) as stream:
    outputs = model(**inputs)

print(stream)

# %%
# AVE st

SEED = 0
# freq_penalty isn't a thing in huggingface
sampling_kwargs = dict(temperature=1.0, top_p=0.3) # , freq_penalty=1.0)
sampling_kwargs['repetition_penalty'] = 2.

# TODO: Automatic padding (easy)
prompt_add, prompt_sub = 'Love ', 'Hate'
coeff = 5
act_name = 6
prompt = "I hate you because"

# %%
# Get activations. Could be made parallel, but this is fine for now.

def get_resid_pre(prompt: str, layer: int):
    with residual_stream(model, layers=[layer]) as stream:
        outputs = model(**tokenize(prompt))
    return stream[layer]

act_add = get_resid_pre(prompt_add, act_name)
act_sub = get_resid_pre(prompt_sub, act_name)
assert act_add.shape == act_sub.shape

act_diff = act_add - act_sub

# %%
# Generate text from modified model

# seed all the things
t.manual_seed(SEED)
np.random.seed(SEED)


def _hook(_, inp):
    resid_pre, = inp
    if resid_pre.shape[1] == 1:
        return # caching in model.generate for new tokens
    
    # We only add to the prompt (first call), not the generated tokens.
    ppos, apos = resid_pre.shape[1], act_diff.shape[1]
    assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"

    # add to the beginning (position-wise) of the activations
    # TODO: Figure out why coeff=5 doesn't work. It works with transformerlens!
    resid_pre[:, :apos, :] += 1 * act_diff


layer = get_blocks(model)[act_name]
with pre_hooks(hooks=[(layer, _hook)]):
    outputs = model.generate(
        **tokenize([prompt] * 5),
        generation_config=GenerationConfig(
            **sampling_kwargs,
            do_sample=True,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
        ),
    )


res_strs = [tokenizer.decode(o) for o in outputs]
print(("\n\n" + "-" * 80 + "\n\n").join(res_strs))
# %%
