"""Functions for processing autoencoders into top-k tokens."""


from collections import defaultdict
from math import ceil

import torch as t
from accelerate import Accelerator
from transformers import AutoTokenizer


def project_activations(
    acts_list: list[t.Tensor],
    projector,
) -> list[t.Tensor]:
    """Projects the activations block over to the sparse latent space."""
    projected_activations: list = []

    for question in acts_list:
        proj_question: list = []
        for activation in question:
            # Detach the gradients from the decoder model pass.
            proj_question.append(projector(activation).detach())

        question_block = t.stack(proj_question)

        projected_activations.append(question_block)

    return projected_activations


def calculate_effects(
    question_token_ids: list[list[int]],
    feature_activations: list[t.Tensor],
    model,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    batch_size: int,
) -> defaultdict[int, defaultdict[str, float]]:
    """Calculate the per input token activation for each feature."""
    number_batches = ceil(len(feature_activations) / batch_size)
    print(f"Number of batches: {number_batches}")

    def new_defaultdict():
        return defaultdict(str)

    neuron_token_effects = defaultdict(new_defaultdict)

    flat_ids: list[int] = [
        token_id for question in question_token_ids for token_id in question
    ]
    tensorized_ids: t.Tensor = t.tensor(flat_ids).to(
        model.encoder_layer.weight.device
    )
    tensorized_ids: t.Tensor = accelerator.prepare(tensorized_ids)
    # Deduplicate token ids.
    set_ids: list[int] = list(set(flat_ids))

    start_point = 0
    end_point = 0
    for batch_index in range(number_batches):
        start_index = batch_index * batch_size
        end_index = (batch_index + 1) * batch_size

        batch_slice = feature_activations[start_index:end_index]
        batch_slice = [accelerator.prepare(tensor) for tensor in batch_slice]
        # Final `batch_slice.shape = (batch_size, projection_dim)`
        batch_slice = t.cat(batch_slice, dim=0).to(
            model.encoder_layer.weight.device
        )
        batch_slice = accelerator.prepare(batch_slice)

        end_point += len(batch_slice)
        for token_id in set_ids:
            fancy_index: t.Tensor = (tensorized_ids == token_id)[
                start_point:end_point
            ]
            id_activations: t.Tensor = batch_slice[fancy_index]

            # Sum along the number of instances (dim=0).
            if len(id_activations) > 0:
                average_activation = t.mean(id_activations, dim=0)
            token_string = tokenizer.convert_ids_to_tokens(token_id)

            for neuron, activation in enumerate(average_activation):
                neuron_token_effects[neuron][token_string] = activation.item()

        start_point = end_point
        print(f"Batch {batch_index+1} of {number_batches} complete.")

    return neuron_token_effects
