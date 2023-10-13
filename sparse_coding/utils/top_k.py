"""Functions for processing autoencoders into top-k tokens."""


from collections import defaultdict
from math import ceil

import torch as t
from accelerate import Accelerator
from transformers import AutoTokenizer


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
    print(f"Total number of batches to be run: {number_batches}")
    print("Starting pre-processing.")

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
    print("Pre-processing complete!")
    for batch_index in range(number_batches):
        print(f"Starting batch {batch_index+1} of {number_batches}.")
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
        print(f"Batch {batch_index+1} of {number_batches} complete!")

    return neuron_token_effects


def project_activations(
    acts_list: list[t.Tensor],
    projector,
    accelerator: Accelerator,
) -> list[t.Tensor]:
    """Projects the activations block over to the sparse latent space."""

    # Remember the original question lengths.
    lengths: list[int] = [len(question) for question in acts_list]

    flat_acts: t.Tensor = t.cat(acts_list, dim=0)
    flat_acts: t.Tensor = accelerator.prepare(flat_acts)
    projected_flat_acts: t.Tensor = projector(flat_acts).detach()

    # Reconstruct the original question lengths.
    projected_activations: list[t.Tensor] = []
    current_idx: int = 0
    for length in lengths:
        projected_activations.append(
            projected_flat_acts[current_idx : current_idx + length, :]
        )
        current_idx += length

    return projected_activations


def select_top_k_tokens(
    effects_dict: defaultdict[int, defaultdict[str, float]],
    top_k: int,
) -> defaultdict[int, list[tuple[str, float]]]:
    """Select the top-k tokens for each feature."""
    tp_k_tokens = defaultdict(list)

    for feature_dim, tokens_dict in effects_dict.items():
        # Sort tokens by their dimension activations.
        sorted_effects: list[tuple[str, float]] = sorted(
            tokens_dict.items(), key=lambda x: x[1], reverse=True
        )
        # Add the top-k tokens.
        tp_k_tokens[feature_dim] = sorted_effects[:top_k]

    return tp_k_tokens


def unpad_activations(
    activations_block: t.Tensor, unpadded_prompts: list[list[int]]
) -> list[t.Tensor]:
    """
    Unpads activations to the lengths specified by the original prompts.

    Note that the activation block must come in with dimensions (batch x stream
    x embedding_dim), and the unpadded prompts as an array of lists of
    elements.
    """
    unpadded_activations: list = []

    for k, unpadded_prompt in enumerate(unpadded_prompts):
        try:
            original_length: int = len(unpadded_prompt)
            # From here on out, activations are unpadded, and so must be
            # packaged as a _list of tensors_ instead of as just a tensor
            # block.
            unpadded_activations.append(
                activations_block[k, :original_length, :]
            )
        except IndexError:
            print(f"IndexError at {k}")
            # This should only occur when the data collection was interrupted.
            # In that case, we just break when the data runs short.
            break

    return unpadded_activations
