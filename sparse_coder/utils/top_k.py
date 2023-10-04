"""Functions for processing autoencoders into top-k tokens."""


from collections import defaultdict
from math import ceil

import torch as t
from accelerate import Accelerator
from transformers import AutoTokenizer


# Calculate per-input-token summed activation, for each feature dimension.
def calculate_effects(
    question_token_ids: list[list[int]],
    feature_activations: list[t.Tensor],
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    batch_size: int,
) -> defaultdict[int, defaultdict[str, float]]:
    """Calculate the per input token activation for each feature."""
    number_batches = ceil(len(feature_activations) / batch_size)

    def new_defaultdict():
        return defaultdict(str)

    neuron_token_effects = defaultdict(new_defaultdict)

    flat_ids: list[int] = [
        token_id for question in question_token_ids for token_id in question
    ]

    tensorized_ids: t.Tensor = accelerator.prepare(t.tensor(flat_ids))
    # Deduplicate token ids.
    set_ids: list[int] = list(set(flat_ids))

    start_point = 0
    end_point = 0
    for batch_index in range(number_batches):
        start_index = batch_index * batch_size
        end_index = (batch_index + 1) * batch_size

        batch_slice = feature_activations[start_index:end_index]
        batch_slice = [accelerator.prepare(tensor) for tensor in batch_slice]
        # Final `batch_slice.shape = (num_batch_activations, PROJECTION_DIM)`
        batch_slice = accelerator.prepare(t.cat(batch_slice, dim=0))

        end_point += len(batch_slice)
        for i in set_ids:
            mask: t.Tensor = (tensorized_ids == i)[start_point:end_point]
            masked_activations: t.Tensor = batch_slice[mask]

            # Sum along the number of instances (dim=0).
            average_activation = t.mean(masked_activations, dim=0)
            token_string = tokenizer.convert_ids_to_tokens(i)

            for neuron, activation in enumerate(average_activation):
                neuron_token_effects[neuron][token_string] = activation.item()

        start_point = end_point

    return neuron_token_effects
