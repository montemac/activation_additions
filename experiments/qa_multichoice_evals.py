# %%
"""The TruthfulQA multiple-choice evals on `Llama-2` models.

Requires a HuggingFace access token for the `Llama-2` models.
"""


import numpy as np
import torch as t
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from numpy import ndarray
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


assert (
    transformers.__version__ == "4.31.0"
), "Llama-2 70B requires at least transformers v4.31.0"

# %%
# NOTE: Don't commit your HF access token!
HF_ACCESS_TOKEN: str = ""
MODEL_DIR: str = "meta-llama/Llama-2-7b-hf"
SEED: int = 0
MAX_NEW_TOKENS: int = 50
NUM_RETURN_SEQUENCES: int = 1
NUM_SHOT: int = 6
NUM_DATAPOINTS: int = 25  # Number of questions evaluated.

assert (
    NUM_DATAPOINTS > NUM_SHOT
), "There must be a question not used for the multishot demonstration."
# %%
# Reproducibility.
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Efficient inference and model parallelization.
t.set_grad_enabled(False)
accelerator: Accelerator = Accelerator()
# `device_map="auto` helps initialize big models.
model: PreTrainedModel = AutoModel.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    use_auth_token=HF_ACCESS_TOKEN,
)

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_auth_token=HF_ACCESS_TOKEN,
)
# The `prepare` wrapper takes over parallelization from here on.
model: PreTrainedModel = accelerator.prepare(model)
model.eval()

# %%
# Load the TruthfulQA dataset.
dataset: dict = load_dataset("truthful_qa", "multiple_choice")

assert (
    len(dataset["validation"]["question"]) >= NUM_DATAPOINTS
), "More datapoints sampled than exist in the dataset."

random_indices: ndarray = np.random.choice(
    len(dataset["validation"]["question"]),
    size=NUM_DATAPOINTS,
    replace=False,
)

random_indices: list = random_indices.tolist()

# %%
# The model answers questions on the `multiple-choice 1` task.
for r in random_indices:
    print(dataset["validation"]["question"][r])
    print(dataset["validation"]["mc1_targets"][r]["choices"])
    print(dataset["validation"]["mc1_targets"][r]["labels"])
