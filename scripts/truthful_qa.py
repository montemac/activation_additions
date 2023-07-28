# %%
import torch as t
import transformers

assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B needs at least transformers 4.31.0."

from datasets import load_dataset
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
dataset = load_dataset("domenicrosati/TruthfulQA")
