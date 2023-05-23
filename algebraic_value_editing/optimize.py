"""Module implenting activation addition optimization."""
from typing import Optional, Any, Iterable

from tqdm.auto import tqdm
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from jaxtyping import Int, Float
import wandb
from transformer_lens import HookedTransformer


def corpus_to_token_batches(
    model: HookedTransformer,
    filenames: Optional[dict[str, Iterable[str]]] = None,
    texts: Optional[dict[str, Iterable[str]]] = None,
    context_len: int = 32,
    stride: int = 4,
):
    """Function to load, tokenize and batch up labeled input texts."""
    assert (filenames is not None) + (
        texts is not None
    ) == 1, "Exactly one of filenames and texts must be provided"

    # Process inputs
    if filenames is not None:
        do_load_text = True
        inputs_by_label = filenames
    else:
        do_load_text = False
        inputs_by_label = texts

    # Create datasets
    tokens_by_label = {}
    for label, texts_or_filenames in inputs_by_label.items():
        texts = []
        for text_or_filename in texts_or_filenames:
            if do_load_text:
                with open(text_or_filename, "r", encoding="utf8") as file:
                    text = file.read()
            else:
                text = text_or_filename
            texts.append(text)
        text = model.tokenizer.eos_token.join(texts)
        tokens = model.to_tokens(text)
        inds = (
            t.arange(context_len)[None, :]
            + t.arange(0, tokens.shape[1] - context_len, stride)[:, None]
        )
        token_snippets = tokens[0, :][inds]
        tokens_by_label[label] = token_snippets

    return tokens_by_label


class AlignedTokensDataset(Dataset):
    """Dataset that stores sequences of tokenized text with associated
    "is aligned" information, and optionally includes pre-cached losses
    calculated using a provided model."""

    def __init__(
        self, tokens_by_label, aligned_labels, model=None, batch_size=10
    ):
        """Initialize a dataset of aligned and non-aligned tokens sequences."""
        # Iterate over labels, adding token batch tensors and is_aligned
        # tensors to list as we go
        tokens_list = []
        is_aligned_list = []
        for label, tokens in tokens_by_label.items():
            tokens_list.append(tokens)
            is_aligned_list.append(
                t.full_like(tokens[:, 0], label in aligned_labels, dtype=bool)
            )
        self.tokens = t.concat(tokens_list, dim=0)
        self.is_aligned = t.concat(is_aligned_list, dim=0)
        assert (
            self.tokens.shape[0] == self.is_aligned.shape[0]
        ), "Tokens and is_aligned shape mismatch"

        # Calculate and cache loss if model is provided
        if model is not None:
            with t.no_grad():
                normal_losses_list = []
                for start_idx in tqdm(
                    range(0, self.tokens.shape[0], batch_size)
                ):
                    tokens_batch = self.tokens[
                        start_idx : (start_idx + batch_size), :
                    ]
                    loss_per_token = model(
                        tokens_batch,
                        return_type="loss",
                        loss_per_token=True,
                    )
                    normal_losses_list.append(loss_per_token)
                self.normal_loss = t.concat(normal_losses_list, dim=0)
                assert (
                    self.tokens.shape[0] == self.normal_loss.shape[0]
                ), "Tokens and normal_loss shape mismatch"
        else:
            self.normal_loss = None

    def __len__(self):
        """Return size of batch dimension"""
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        """Get specific items by index (returns tokens and is_aligned)"""
        items = {
            "tokens": self.tokens[idx, :],
            "is_aligned": self.is_aligned[idx],
        }
        if self.normal_loss is not None:
            items["normal_loss"] = self.normal_loss[idx, :]
        return items


def learn_activation_addition(
    model: HookedTransformer,
    tokens_by_label: dict[str, Int[t.Tensor, "batch pos"]],
    aligned_labels: list[str],
    act_name: str,
    unaligned_loss_method: str = "abs_of_mean",
    lr: float = 0.01,
    weight_decay: float = 0.01,
    num_epochs: int = 100,
    batch_size: int = 20,
    seed: int = 0,
    do_print: bool = True,
    use_wandb: bool = False,
    wandb_project_name: Optional[str] = None,
    wandb_additional_config: Optional[dict[str, Any]] = None,
) -> nn.Parameter:
    """Function to learn an activation addition vector (aka steering
    vector) over a specific set of labelled inputs."""
    assert unaligned_loss_method in ["abs_of_mean", "mean_of_abs"]
    # Set up logging, if provided
    if use_wandb:
        if wandb_project_name is None:
            wandb_project_name = "learning_activation_additions"
        wandb_config = {
            "model_cfg": model.cfg,
            "token_labels": list(tokens_by_label.keys()),
            "aligned_labels": aligned_labels,
            "act_name": act_name,
            "unaligned_loss_method": unaligned_loss_method,
            "lr": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "seed": seed,
        }
        if wandb_additional_config is not None:
            wandb_config.update(wandb_additional_config)
        wandb.init(
            project=wandb_project_name,
            config=wandb_config,
        )

    # Create the dataset
    dataset = AlignedTokensDataset(
        tokens_by_label, aligned_labels, model=model, batch_size=batch_size
    )

    # Create the steering vector parameter, and an associated hook
    # function
    steering_vector = nn.Parameter(
        t.randn(model.cfg.d_model, device=model.cfg.device),
        requires_grad=True,
    )

    def hook_fn(activation, hook):  # pylint: disable=unused-argument
        """Hook function"""
        activation[:, 0, :] += steering_vector
        return activation

    # Create an optimizer
    optimizer = t.optim.AdamW(
        [steering_vector],
        lr=lr,
        weight_decay=weight_decay,
    )

    # Create a dataloader
    generator = t.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    # Iterate over epochs, with hook applied via context manager
    with model.hooks(fwd_hooks=[(act_name, hook_fn)]):
        for epoch in tqdm(range(1, num_epochs + 1)):
            epoch_loss = 0.0
            batch_cnt = 0
            for batch in dataloader:
                loss_per_token = model(
                    batch["tokens"], return_type="loss", loss_per_token=True
                )
                relative_loss = loss_per_token - batch["normal_loss"]
                if unaligned_loss_method == "abs_of_mean":
                    loss = (
                        relative_loss[batch["is_aligned"], :].sum()
                        + t.abs(relative_loss[~batch["is_aligned"], :].sum())
                    ) / relative_loss.numel()
                else:  # mean_of_abs
                    loss = (
                        relative_loss * batch["is_aligned"][:, None]
                        + t.abs(relative_loss)
                        * (~batch["is_aligned"][:, None])
                    ).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
                if use_wandb:
                    wandb.log({"loss": loss.item(), "epoch": epoch})
            if do_print:
                print(f"Epoch: {epoch}, Loss: {epoch_loss/batch_cnt}")

    # Don't need grad any more after this!
    steering_vector.requires_grad_(False)

    detached_steering_vector = steering_vector.detach()

    if use_wandb:
        artifact = wandb.Artifact(
            "learned_steering_vector",
            type="activation_addition",
            metadata={"act_name": act_name},
        )
        with open("tmp.pt", "wb") as file:
            t.save(detached_steering_vector, file)
        artifact.add_file("tmp.pt", name="steering_vector")
        wandb.log_artifact(artifact)

    return detached_steering_vector
