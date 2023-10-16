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
    small_model_mode: bool,
) -> defaultdict[int, defaultdict[str, float]]:
    """Calculate the per input token activation for each feature."""

    number_batches = ceil(len(feature_activations) / batch_size)
    print(f"Total number of batches to be run: {number_batches}")
    print("Starting pre-processing.")

    def new_defaultdict():
        return defaultdict(str)

    neuron_token_effects = defaultdict(new_defaultdict)

    flat_input_ids: list[int] = [
        token_id for question in question_token_ids for token_id in question
    ]
    # Deduplicate token ids.
    set_of_ids: list[int] = list(set(flat_input_ids))

    if small_model_mode is True:
        flat_input_ids: t.Tensor = t.tensor(flat_input_ids).to(
            model.encoder_layer.weight.device
        )
    flat_input_ids: t.Tensor = accelerator.prepare(flat_input_ids)

    start_idx = 0
    end_idx = 0
    print("Pre-processing complete!")

    for batch_idx in range(number_batches):
        print(f"Starting batch {batch_idx+1} of {number_batches}.")

        start_index = batch_idx * batch_size
        end_index = (batch_idx + 1) * batch_size

        print(f"feature_activations[0] shape: {feature_activations[0].shape}")

        batch = feature_activations[start_index:end_index]
        batch = [accelerator.prepare(tensor) for tensor in batch]
        # Final `batch.shape = (batch_size, projection_dim)`
        batch = t.cat(batch, dim=0).to(model.encoder_layer.weight.device)
        batch = accelerator.prepare(batch)

        end_idx += len(batch)

        for input_id in set_of_ids:
            token_string = tokenizer.convert_ids_to_tokens(input_id)

            fancy_index: t.Tensor = t.nonzero(
                (flat_input_ids == input_id)[start_idx:end_idx]
            )
            activations_at_input: t.Tensor = batch[fancy_index]

            # Average over number of token activation instances.
            activations_at_input = t.mean(activations_at_input, dim=0)

            activations_at_input = activations_at_input.squeeze(dim=0)

            assert activations_at_input.shape == (
                model.encoder_layer.weight.shape[0],
            ), f"`activations_at_input` length: {activations_at_input.shape} != projection_dim: {model.encoder_layer.weight.shape[0]}"

            for dim, activation in enumerate(activations_at_input):
                neuron_token_effects[dim][token_string] = activation.item()

        start_idx = end_idx
        print(f"Batch {batch_idx+1} of {number_batches} complete!")

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
