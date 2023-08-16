# %%
"""Dict learning on an activations dataset, with a variational autoencoder."""


import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


# %%
# Training hyperparameters. We want to weight L1 extremely heavily.
LAMBDA_L1: float = 1.0
LAMBDA_KL: float = 1e-9

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
        # The first linear layer learns a matrix map to a higher-dimensional
        # space. That projection matrix is what I'm intersted in here.
        self.encoder = t.nn.Sequential(
            t.nn.Linear(4096, 8192),
            t.nn.ReLU(),
            t.nn.Linear(8192, 8192 * 2)
        )

        # The second linear map just returns us to the original activation
        # space, so we can evaluate our reconstruction loss.
        self.decoder = t.nn.Sequential(
            t.nn.Linear(8192, 4096),
        )


    def forward(self, state):  # pylint: disable=arguments-differ
        """The forward pass of a variational autoencoder for activations."""
        encoded_state = self.encoder(state)
        mean, logvar = encoded_state.split(8192, dim=-1)

        # Sample from the encoder normal distribution.
        std = t.exp(0.5 * logvar)  # pylint: disable=no-member
        epsilon = t.randn_like(std)  # pylint: disable=no-member
        sampled_state = mean + (epsilon * std)

        # Decode the sampled state.
        output_state = self.decoder(sampled_state)
        return mean, logvar, sampled_state, output_state


    def training_step(self, batch):  # pylint: disable=arguments-differ
        """Train the autoencoder."""
        state = batch
        mean, logvar, sampled_state, output_state = self.forward(state)

        # For the statistical component of the forward pass.
        kl_loss = -0.5 * t.sum(1 + logvar - mean.pow(2) - logvar.exp())  # pylint: disable=no-member

        # I want to learn a _sparse representation_ of the original learned
        # features present in the training activations. L1 regularization in the
        # higher-dimensional space does this.
        l1_loss = t.nn.functional.l1_loss(sampled_state, t.zeros_like(sampled_state))  # pylint: disable=no-member

        # I also need to inventivize learning features that match the originals.
        # I project back to the original space, then evaluate MSE for this.
        mse_loss = t.nn.functional.mse_loss(output_state, state)

        # The overall loss function just combines the three above terms.
        loss = mse_loss + (LAMBDA_L1 * l1_loss) + (LAMBDA_KL * kl_loss)

        self.log("loss", loss)

        self.log("L1 sub-loss", l1_loss)
        self.log("KL sub-loss", kl_loss)
        self.log("MSE sub-loss", mse_loss)
        return loss


    def configure_optimizers(self):
        """Configure the optimizer (`Adam`)."""
        return t.optim.Adam(self.parameters(), lr=1e-4)


# %%
# Train the autoencoder.
model: Autoencoder = Autoencoder()
trainer: pl.Trainer = pl.Trainer(accelerator="auto", max_epochs=150, log_every_n_steps=1)

trainer.fit(model, dataloader)

# %%
# Save the learned projection matrix.
t.save(
    model.encoder[0].weight.data, "/root/algebraic_value_editing/experiments/projection_matrix.pt"
)
