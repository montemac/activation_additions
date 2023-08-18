# %%
"""Sort feature directions in the decoder by impact on `truthful_qa` score."""


import csv
from contextlib import contextmanager

import numpy as np
import torch as t
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from numpy import ndarray
from transformers import (
    AutoModelForCausalLM,
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
DECODER_PATH: str = "acts_data/learned_decoder.pt"
IMPACT_SAVE_PATH: str = "acts_data/impacts.csv"
SEED: int = 0
INJECTION_LAYER: int = 16  # Layer to _add_ feature directions to.
COEFF: float = 2.0  # Coefficient for the feature addition.
MAX_NEW_TOKENS: int = 1
NUM_RETURN_SEQUENCES: int = 1
NUM_SHOT: int = 6
NUM_DATAPOINTS: int = 20  # Number of questions evaluated.

assert (
    NUM_DATAPOINTS > NUM_SHOT
), "There must be a question not used for the multishot demonstration."

# %%
# Reproducibility.
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Efficient inference and model parallelization.
# `device_map="auto"` helps to initialize big models.
t.set_grad_enabled(False)
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    use_auth_token=HF_ACCESS_TOKEN,
)

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_auth_token=HF_ACCESS_TOKEN,
)

accelerator: Accelerator = Accelerator()
model: PreTrainedModel = accelerator.prepare(model)
model.eval()

# %%
# Load the TruthfulQA multichoice dataset.
dataset: dict = load_dataset("truthful_qa", "multiple_choice")

assert (
    len(dataset["validation"]["question"]) >= NUM_DATAPOINTS
), "More datapoints sampled than exist in the dataset."

sampled_indices: ndarray = np.random.choice(
    len(dataset["validation"]["question"]),
    size=NUM_DATAPOINTS,
    replace=False,
)

sampled_indices: list = sampled_indices.tolist()


# %%
# Process the decoder for later addition.
decoder: t.Tensor = t.load(DECODER_PATH)

# Print the ordered basis magnitudes.
magnitudes: list = []

for i in range(decoder.size(1)):
    column: t.Tensor = decoder[:, i]
    col_mag: float = t.norm(column, p=1).item()
    magnitudes.append(col_mag)
magnitudes.sort()

for m in magnitudes:
    print(m)

# Print the ordered basis number of zeros.
num_zeros: list = []

for i in range(decoder.size(1)):
    column: t.Tensor = decoder[:, i]
    col_zeros: int = t.sum(t.abs(column) < 1e-4).item()  # pylint: disable=no-member
    num_zeros.append(col_zeros)
num_zeros.sort()

for n in num_zeros:
    print(n)

# Get projected columns from the decoder.
feature_vectors: list[t.Tensor] = []

for i in range(decoder.size(1)):
    column: t.Tensor = decoder[:, i]
    feature_vectors.append(column)


# %%
# Add in the feature vectors during a pass.
@contextmanager
def register_feat_hooks(feature_vec: t.Tensor, coeff: float, mod: PreTrainedModel, layer_num: int):
    """Add scaled feature vectors into the model's activation space."""

    def hook(module, input_, output):
        return (output[0] + coeff * feature_vec,)

    _hook = mod.model.layers[layer_num].register_forward_hook(hook)
    try:
        yield
    finally:
        _hook.remove()


# %%
# Convert one-hot labels to int indices.
def unhot(labels: list) -> int:
    """Change the one-hot ground truth labels to a 1-indexed int."""
    return np.argmax(labels) + 1


# %%
# The basic evals loop.
def mc_evals(samples_nums: list, num_shot: int = NUM_SHOT) -> dict[int, float]:
    """
    Run the eval loop, collecting the logit on ground truth for each question.
    
    This function counts on the script's `accelerator`, `dataset`, `model`,
    `tokenizer`, and `unhot` variables. It is not designed to work outside of
    this script.
    """
    ground_truth_logits: dict = {}

    for question_num in samples_nums:
        multishot: str = ""
        # Sample multishot questions that _aren't_ the current question.
        multishot_indices: ndarray = np.random.choice(
            [
                x
                for x in range(len(dataset["validation"]["question"]))
                if x != question_num
            ],
            size=num_shot,
            replace=False,
        )

        # Build the multishot prompt.
        for mult_num in multishot_indices:
            multishot += "Q: " + dataset["validation"]["question"][mult_num] + "\n"

            for choice_num in range(
                len(dataset["validation"]["mc1_targets"][mult_num]["choices"])
            ):
                # choice_num is 0-indexed, but I want to display _1-indexed_
                # options.
                multishot += (
                    "("
                    + str(choice_num + 1)
                    + ") "
                    + dataset["validation"]["mc1_targets"][mult_num]["choices"][
                        choice_num
                    ]
                    + "\n"
                )

            labels_one_hot: list = dataset["validation"]["mc1_targets"][mult_num][
                "labels"
            ]
            # Get a label int from the `labels` list.
            correct_answer: int = unhot(labels_one_hot)
            # Add on the correct answer under each multishot question.
            multishot += "A: (" + str(correct_answer) + ")\n"

        # Add on the current question.
        question: str = (
            "Q: " + dataset["validation"]["question"][question_num] + "\n"
        )
        for option_num in range(
            len(dataset["validation"]["mc1_targets"][question_num]["choices"])
        ):
            # option_num is similarly 0-indexed, but I want 1-indexed options
            # here too.
            question += (
                "("
                + str(option_num + 1)
                + ") "
                + dataset["validation"]["mc1_targets"][question_num]["choices"][
                    option_num
                ]
                + "\n"
            )
        # I only want the model to actually answer the question, with a single
        # token, so I tee it up here with the opening parentheses to a
        # multiple-choice answer integer.
        question += "A: ("

        # Tokenize, prepare the model input.
        input_ids: t.Tensor = tokenizer.encode(
            multishot + question, return_tensors="pt"
        )
        input_ids = accelerator.prepare(input_ids)

        # Generate a single-token completion.
        outputs = model(input_ids)

        # Get the ground truth logit. The ground truth is ultimately stored as a
        # string literal, so I have to work a bit to get its logit from the
        # model's final logits.
        ground_truth_one_hot: list = dataset["validation"]["mc1_targets"][question_num][
        "labels"
        ]
        ground_truth_ans: int = unhot(ground_truth_one_hot)
        ground_truth_id: list[int] = tokenizer.encode(str(ground_truth_ans))
        ground_truth_logit: float = outputs.logits[0, -1, ground_truth_id[0]]
        ground_truth_logits[question_num] = ground_truth_logit.item()

    return ground_truth_logits


# %%
# The baseline model's logits for the correct answer.
baseline_ground_truth_logits: dict[int, float] = mc_evals(sampled_indices)

# %%
# The steered model's logits for the correct answer, for each feature vec.
feature_vec_sweeps: dict[int, dict[int, float]] = {}

for i, feature in enumerate(feature_vectors):
    # HACK: Device mismatch between feature and ouput[0].
    feature = feature.to("cuda:3")
    with register_feat_hooks(feature, COEFF, model, INJECTION_LAYER):
        feature_vec_sweeps[i] = mc_evals(sampled_indices)

# %%
# Measure the impact of each feature vector on the correct logit.
final_results: dict[int, float] = {}

for k, feature in feature_vec_sweeps:
    dim_index: int = k
    feature_impact = baseline_ground_truth_logits[k] - feature[k]
    final_results[dim_index] = feature_impact

# %%
# Save the final results.
with open(IMPACT_SAVE_PATH, "a", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Dimension index", "Feature impact on ground truth logit"])
    for d, f in final_results.items():
        writer.writerow([d, f])
    # Add a final newline.
    writer.writerow([])
