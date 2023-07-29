# %%
"""WIP TruthfulQA evals on `Llama-2`."""
import numpy as np
import torch as t
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from datasets import load_dataset
from accelerate import Accelerator

from bleurt import score

assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B needs at least transformers 4.31.0."


# %%
# NOTE: Don't commit HF tokens!
ACCESS_TOKEN: str = ""
MODEL_DIR: str = "meta-llama/Llama-2-7b-chat-hf"
SEED: int = 0
MAX_LENGTH: int = 128
NUM_RETURN_SEQUENCES: int = 1

# %%
# Reproducibility.
t.manual_seed(SEED)
np.random.seed(SEED)

# Efficient inference and model parallelization.
t.set_grad_enabled(False)
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

model: PreTrainedModel = accelerator.prepare(model)
tokenizer: PreTrainedTokenizer = accelerator.prepare(tokenizer)
model.eval()
model.tie_weights()

# %%
# Sample from the TruthfulQA dataset.
dataset = load_dataset("domenicrosati/TruthfulQA")
random_indices = np.random.choice(
    len(dataset["train"]["Question"]), size=10, replace=False
)

# %%
# Generate answers to each passed question.
generated_answers = []

for i in random_indices:
    question = dataset["train"]["Question"][i]
    inputs = tokenizer.encode(
        question + tokenizer.eos_token, return_tensors="pt"
    )
    outputs = model.generate(
        inputs,
        max_length=MAX_LENGTH,
        num_return_sequences=NUM_RETURN_SEQUENCES,
    )
    generated_answers.append(
        tokenizer.decode(outputs[0], skip_special_tokens=True)
    )

# %%
# BLEURT grades the model's answers.
true_refs = [dataset["train"]["Correct Answers"][j] for j in random_indices]
false_refs = [dataset["train"]["Incorrect Answers"][k] for k in random_indices]
scorer = score.BleurtScorer(
    checkpoint="/mnt/ssd-2/mesaoptimizer/david/BLEURT-20"
)
scores_true = []
scores_false = []

for i, answer in enumerate(generated_answers):
    scores_true.append(
        max(
            [
                scorer.score(references=[answer], candidates=[ref])
                for ref in true_refs[i]
            ]
        )
    )

    scores_false.append(
        max(
            [
                scorer.score(references=[answer], candidates=[ref])
                for ref in false_refs[i]
            ]
        )
    )

scores_true = np.array(scores_true)
scores_false = np.array(scores_false)

# %%
# Print the results.
scores_final = scores_true - scores_false
print(f"Final scores: {scores_final}")
