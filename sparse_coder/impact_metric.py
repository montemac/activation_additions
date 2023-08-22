# %%
"""
Evaluate feature directions in the decoder by impact on `truthful_qa` score.

For now, it is set up as a multishot process, but I may refactor it to
zero-shot, to run in reasonable time, especially as I scale up the latent space
size further. Note that you'll need a HuggingFace/Meta access token for the
`Llama-2` models.
"""


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
BATCH_SIZE: int = 20
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
# Load and get projected columns from the decoder.
decoder: t.Tensor = t.load(DECODER_PATH)
feature_vectors: list[t.Tensor] = []

for i in range(decoder.size(1)):
    column: t.Tensor = decoder[:, i]
    feature_vectors.append(column)


# %%
# A hook factory; hooks in `torch` have a fixed type signature.
def hook_factory(feature_vec: t.Tensor, coeff: float):
    """Return a pre-forward hook adding in a scaled feature vector."""

    def _hook(__, _):
        """Add a feature vector to the residual space."""
        # `__` is the model; `_` is the input. The input comes in as and must
        # leave as a tuple.
        residual = _[0]

        # Patch for horrible parallelization bug.
        deviced_feature_vec = feature_vec.to(residual.device)

        # Broadcast the feature vector across the batch, stream dims.
        expanded_feature_vec = deviced_feature_vec.expand(
            residual.size(0), residual.size(1), -1
        )

        assert (
            residual.size() == expanded_feature_vec.size()
        ), f"""
        Model activations and feature vectors do not match in size.
        Model activation size: {residual.size()}
        Feature vector size: {expanded_feature_vec.size()}
        """

        residual += coeff * expanded_feature_vec
        return (residual,)

    return _hook


# %%
# Registering and teardown a hook, for a pass.
@contextmanager
def register_hook(model_layer_module, hook_function):
    """Register (and unregister) a pre-forward hook."""
    handle = model_layer_module.register_forward_pre_hook(hook_function)

    try:
        yield
    finally:
        handle.remove()


# %%
# Convert one-hot labels to int indices.
def unhot(labels: list) -> int:
    """Change the one-hot ground truth labels to a 1-indexed int."""
    return np.argmax(labels) + 1


# %%
# Build multishot prompts.
def build_multishot_prompt(q_num: int, num_shot: int) -> str:
    """
    Build up the multishot prompt for a question index.

    Implicitly relies on `dataset` and `unhot`.
    """
    multishot: str = ""
    # Sample multishot questions that _aren't_ the current question.
    multishot_indices: ndarray = np.random.choice(
        [
            x
            for x in range(len(dataset["validation"]["question"]))
            if x != q_num
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
    question: str = "Q: " + dataset["validation"]["question"][q_num] + "\n"
    for option_num in range(
        len(dataset["validation"]["mc1_targets"][q_num]["choices"])
    ):
        # option_num is similarly 0-indexed, but I want 1-indexed options
        # here too.
        question += (
            "("
            + str(option_num + 1)
            + ") "
            + dataset["validation"]["mc1_targets"][q_num]["choices"][
                option_num
            ]
            + "\n"
        )
    # I only want the model to actually answer the question, with a single
    # token, so I tee it up here with the opening parentheses to a
    # multiple-choice answer integer.
    question += "A: ("

    return multishot + question


# %%
# The basic evals loop.
def mc_evals(
    samples_nums: list, batch_size: int = BATCH_SIZE, num_shot: int = NUM_SHOT
) -> dict[int, float]:
    """
    Run the eval loop, collecting the logit on ground truth for each question.

    This function counts on the script's `accelerator`, `dataset`, `model`,
    `tokenizer`, and `unhot` variables. It is not designed to work outside of
    this script.
    """
    ground_truth_logits: dict = {}

    # Get slice indices.
    for batch_first in range(0, len(samples_nums), batch_size):
        batch_indices: list = samples_nums[
            batch_first : (batch_first + batch_size)
        ]

        batch_inputs: list[t.Tensor] = []
        for question_num in batch_indices:
            prompt: str = build_multishot_prompt(question_num, num_shot)

            # Tokenize, prepare the model input.
            input_ids: t.Tensor = tokenizer.encode(prompt, return_tensors="pt")
            # Remove the batch dim of 1, since we'll rebatch with a new dim
            # shortly.
            batch_inputs.append(input_ids.squeeze())

        # Pad, tensorize, and prepare the batch.
        batch_inputs: t.Tensor = t.nn.utils.rnn.pad_sequence(
            batch_inputs, batch_first=True
        )
        batch_inputs = accelerator.prepare(batch_inputs)

        # Generate a single-token completion for each batched input.
        outputs = model(batch_inputs)

        # Get the ground truth logit. The ground truth is ultimately stored as a
        # string literal, so I have to work a bit to get its logit from the
        # model's final logits.
        for batch_indx, question_num in enumerate(batch_indices):
            ground_truth_one_hot: list = dataset["validation"]["mc1_targets"][
                question_num
            ]["labels"]
            ground_truth_ans: int = unhot(ground_truth_one_hot)
            ground_truth_id: list[int] = tokenizer.encode(
                str(ground_truth_ans)
            )
            ground_truth_logit: float = outputs.logits[
                batch_indx, -1, ground_truth_id[0]
            ]
            ground_truth_logits[question_num] = ground_truth_logit.item()

    return ground_truth_logits


# %%
# The baseline model's logits for the correct answer.
baseline_ground_truth_logits: dict[int, float] = mc_evals(sampled_indices)

# %%
# The steered model's logits for the correct answer, for each feature vec.
feature_vec_sweeps: dict[int, dict[int, float]] = {}

for k, feature_vector in enumerate(feature_vectors):
    with register_hook(
        model.model.layers[INJECTION_LAYER],
        hook_factory(feature_vector, COEFF),
    ):
        feature_vec_sweeps[k] = mc_evals(sampled_indices)

# %%
# Measure the impact of each feature vector on the correct logit.
final_results: dict[int, float] = {}

for direction_index, direction_addition in feature_vec_sweeps.items():
    direction_effect: float = 0.0

    for question_index, modded_logit in direction_addition.items():
        baseline_logit = baseline_ground_truth_logits[question_index]
        question_impact = baseline_logit - modded_logit
        direction_effect += question_impact

    final_results[direction_index] = direction_effect

sorted_results = sorted(
    final_results.items(), key=lambda x: x[1], reverse=True
)

# %%
# Save the final results.
with open(IMPACT_SAVE_PATH, "a", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        ["Dimension index", "Feature impact on ground truth logit"]
    )
    for d, f in sorted_results:
        writer.writerow([d, f])
    # Add a final newline.
    writer.writerow([])
