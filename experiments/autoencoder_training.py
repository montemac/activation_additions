# %%
"""Dictionary learning on an activations dataset using an autoencoder."""


import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


# %%
# Define a `torch` dataset.
class ActivationsDataset(Dataset):
    """Dataset of hidden states from a pretrained model."""

    def __init__(self, path):
        """Constructor; inherits from `torch.utils.data.Dataset` class."""
        self.data = t.load(path)

    def __len__(self):
        """Return the dataset length."""
        return len(self.data)

    def __getitem__(self, indx):
        """Return the item at the passed index."""
        return self.data[indx]


# %%
# Put the dataset into a dataloader.
dataset: ActivationsDataset = ActivationsDataset(
    "/root/algebraic_value_editing/experiments/activations_dataset.pt"
    )

dataloader: DataLoader = DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=1,
)


# %%
# Define a variational autoencoder with `lightning`.
class Autoencoder(pl.LightningModule):
    """A variational autoencoder architecture."""

    def __init__(self):
        super().__init__()
        self.encoder = t.nn.Sequential(
            t.nn.Linear(4096, 256),
            t.nn.ReLU(),
            t.nn.Linear(256, 64),
            t.nn.ReLU(),
            t.nn.Linear(64, 20),
            t.nn.ReLU(),
        )
        self.decoder = t.nn.Sequential(
            t.nn.Linear(20, 64),
            t.nn.ReLU(),
            t.nn.Linear(64, 256),
            t.nn.ReLU(),
            t.nn.Linear(256, 8192),
            # TODO: How should the higher-dim output be trained on?
            t.nn.Linear(8192, 4096),
        )

    def forward(self, state):  # pylint: disable=arguments-differ
        """The forward pass of the autoencoder. Information is compacted, then reconstructed."""
        state = self.encoder(state)
        state = self.decoder(state)
        return state

    def training_step(self, batch):  # pylint: disable=arguments-differ
        """Train the autoencoder."""
        datapoint = batch
        encoded_state = self.encoder(datapoint)
        prediction = self.forward(datapoint)

        # Reconstruction loss.
        mse_loss = t.nn.functional.mse_loss(prediction, datapoint)

        # Sparsity loss.
        l1_loss = t.nn.functional.l1_loss(encoded_state, t.zeros_like(encoded_state))

        # Total loss.
        lambda_l1: float = 1e-5
        loss = mse_loss + lambda_l1 * l1_loss

        self.log("training_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer (`Adam`)."""
        return t.optim.Adam(self.parameters(), lr=1e-3)


# %%
# Train the autoencoder.
model: Autoencoder = Autoencoder()

trainer: pl.Trainer = pl.Trainer(accelerator="auto", max_epochs=50)
trainer.fit(model, dataloader)
