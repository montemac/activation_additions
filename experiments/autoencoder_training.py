# %%
"""Dictionary learning on an activations dataset using a variational autoencoder."""


import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


# %%
# Use available tensor cores.
t.set_float32_matmul_precision("high")


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
    dataset, batch_size=32, shuffle=True, num_workers=48,
)


# %%
# Define a variational autoencoder with `lightning`.
class Autoencoder(pl.LightningModule):
    """A variational autoencoder architecture."""

    def __init__(self):
        super().__init__()
        # TODO: Add in the _variational_ component. The first linear layer
        # learns a matrix map to a higher-dimensional space. That projection
        # matrix is what I'm intersted in here.
        self.encoder = t.nn.Sequential(
            t.nn.Linear(4096, 8192),
            t.nn.ReLU(),
        )

        # The second linear map returns us to the original activation space, so
        # we can evaluate our reconstruction loss.
        self.decoder = t.nn.Sequential(
            t.nn.Linear(8192, 4096),
        )

    def forward(self, state):  # pylint: disable=arguments-differ
        """The forward pass of the autoencoder. Information is compacted, then reconstructed."""
        encoded_state = self.encoder(state)
        output_state = self.decoder(encoded_state)
        return encoded_state, output_state

    def training_step(self, batch):  # pylint: disable=arguments-differ
        """Train the autoencoder."""
        state = batch
        encoded_state, output_state = self.forward(state)

        # I want to learn a disentangled, sparse representation of the original
        # learned features present in the training activations. L1
        # regularization in the higher-dimensional space does this.
        l1_loss = t.nn.functional.l1_loss(encoded_state, t.zeros_like(encoded_state))  # pylint: disable=no-member

        # I also need to inventivize learning features that match the originals.
        # I project back to the original space, then evaluate MSE for this.
        mse_loss = t.nn.functional.mse_loss(output_state, state)

        # The total loss function just combines the two above terms.
        lambda_l1: float = 1e-1
        loss = mse_loss + lambda_l1 * l1_loss
        
        # TODO: Log sparsity data along with training loss.
        self.log("training_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer (`Adam`)."""
        return t.optim.Adam(self.parameters(), lr=1e-3)


# %%
# Train the autoencoder.
model: Autoencoder = Autoencoder()

trainer: pl.Trainer = pl.Trainer(accelerator="auto", max_epochs=150, log_every_n_steps=1)
trainer.fit(model, dataloader)
