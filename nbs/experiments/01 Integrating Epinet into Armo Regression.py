# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: rmurdo-zetteldev
#     language: python
#     name: rmurdo-zetteldev
# ---

# %%
#|hide
## Standard libraries
import os
import math
import numpy as np
import time
from fastcore.all import *
from nbdev.showdoc import *

## Imports for plotting
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

## Progress bar
from tqdm.auto import tqdm, trange

## project specifics
import murdo
import transformers

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Integrating Epinet Into Armo Regression
# > C'mon, c'mon and meet the epinets

# %% [markdown]
# ArmoRM's regression layer is pretty simple: it's just a linear layer, with a bias term, optimized through Ridge regression. In other words, it's just performing a change of basis on the underlying last-layer embeddings of Llama3 8b. And apparently, dimensions already exist in this marvelous latent space that correspond to many of the dimensions we care about.
#
# What can the epinet add to this? First, the structure fo the epinet should be pretty simple: we'll replicate the linear layer being used with Ridge Regression an MLP (hoping gradient descent is sufficient to extract the same dimensional information; and assuming we need extra compute power to both get the dimensions and reason about their epistemic status), but will multiply the output by the random epistemic index. In theory, this should allow the epinet to reduce the MSE regression loss by adding randomness to dimensions with questionable epistemic status.
#
# In this notebook, we integrate an mlp epinet with the regression layer. We'll try the simplest possible integration first, then perform training (which, given the small size, should be pretty quick), and iterate.
#
# How will we measure whether the integration works? First, we can sanity check by seeing how uncertainty compares across dimensions of rewards, comparing with our prior data on which dimensions have the most activation and which appear to be duplicates of each other. Ultimately, we can measure the performance of the reward model by doing some uncertainty-weighted best of N search.
#
# A note on form: prior versions of zetteldev had an emphasis on atomic notebooks for experimentation. We break from that. This document is more 'computational essay/lab report' than slip. It will contain many ideas, and confront much computational and ideological reducibility. The metaphors worthy of further abstraction will be highlighted in a separate report, so see the 'Reports' folder for the high level summary. What follows is a 'lab report' in chronological order.

# %% [markdown]
# **Hypothesis**:
# 1. Integrating the MLP reward model with an MLP epinet will enable the prediction of uncertainty per reward dimension per prompt.
# 2. The uncertainty estimate should change with prompt response pairs.
# 3. When the gating layer denotes a reward dimension as irrelevant, it should have a higher uncertainty.
# 4. Integrating uncertainty into a Best of N sampler from a base llama model should have superior performance to the reward model without uncertainty.

# %% [markdown]
# # Machinery

# %% [markdown]
# First, we'll set up the pretrained reward model, then extract dimensions and such from it.

# %%
import torch, numpy

# %%
model_name, dataset_name = ("FsfairX-LLaMA3-RM-v0.1", "ArmoRM-Multi-Objective-Data-v0.1")
save_dir = os.path.join("/home/piriac", "data", "ArmoRM", "regression_weights")
save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}.pt")
regression_layer = torch.load(save_path)["weight"]

# %%
n_attributes, hidden_size = regression_layer.shape

# %% [markdown]
# Example usage:

# %%
pairwise_rewards = torch.rand(800,hidden_size) @ regression_layer.T

# %% [markdown]
# Load the dataset prompt-response embeddings from the base LLM, as inputs to the regression layer.

# %%
import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from safetensors.torch import load_file
from argparse import ArgumentParser
def load_embeddings_and_preferences(embeddings_dir=None, model_name=None, dataset_name=None):
    """
    Load embeddings and preferences from safetensors files.

    Args:
        embeddings_dir (str, optional): Path to embeddings directory
        model_name (str, optional): Name of the model
        dataset_name (str, optional): Name of the dataset

    Returns:
        tuple: (embeddings tensor, labels tensor)
    """
    # Set default paths if not provided
    HOME = os.path.expanduser("~")
    if embeddings_dir is None:
        embeddings_dir = os.path.join(
            HOME, "data", "ArmoRM", "embeddings", model_name, dataset_name
        )

    # Collect all embedding files
    embedding_files = sorted(glob(f"{embeddings_dir}-*.safetensors"))

    if not embedding_files:
        raise FileNotFoundError(f"No embedding files found in {embeddings_dir}")

    # Initialize lists to store embeddings and labels
    embeddings = []
    labels = []

    print("Loading embeddings and labels from Safetensors files...")
    for file in tqdm(embedding_files, desc="Loading embeddings"):
        # Load the safetensors file
        data = load_file(file)
        embeddings.append(data["embeddings"])  # Append embeddings tensor
        labels.append(data["labels"])  # Append labels tensor

    # Concatenate all embeddings and labels into single tensors
    embeddings = torch.cat(embeddings, dim=0).float()
    labels = torch.cat(labels, dim=0).float()

    print(f"Total embeddings loaded: {embeddings.shape[0]}")
    print(f"Total labels loaded: {labels.shape[0]}")

    # Verify shapes match
    assert embeddings.shape[0] == labels.shape[0], "Number of embeddings and labels must match"

    return embeddings, labels


# %%
embeddings, sparse_rewards = load_embeddings_and_preferences(
    model_name="FsfairX-LLaMA3-RM-v0.1",
    dataset_name="ArmoRM-Multi-Objective-Data-v0.1"
)

# %% [markdown]
# Only 15% of the reward labels are present.

# %%
sparse_rewards.shape

# %%
sparse_rewards[sparse_rewards == sparse_rewards].shape

# %%
1647670/(569185*19)

# %%
embeddings

# %% [markdown]
# Sanity check the regression weights by seeing how well it matches the preferences.

# %%
predicted_rewards = embeddings @ regression_layer.T

# %%
~torch.isnan(sparse_rewards)

# %%
diff = (sparse_rewards - predicted_rewards)[~torch.isnan(sparse_rewards)].numpy()
mse_diff = np.mean(diff**2)

# %%
mse_diff

# %% [markdown]
# So the reward prediction is extremely successful.

# %% [markdown]
# ## Setting up the epinet

# %% [markdown]
# For our first epinet, we'll use a two layer mlp for the randomized component, and a pure linear layer for deterministic component. Thus, the non-randomized network recreates the Ridge regression setting. The epinet is given slightly more structure under the intuition that it needs not only to reproduce the computations of the deterministic component, but also reason about when those calculations need added randomness.

# %%
from murdo.epinet_mlp import make_mlp_epinet
epinet, indexer = make_mlp_epinet(
    output_sizes = [hidden_size,n_attributes],
    epinet_hiddens = [hidden_size + n_attributes, 512],
    index_dim = 8,
    prior_scale = 1,
    name = "my first epinet",
)

# %%
epinet

# %%
# example usage
output = epinet(torch.randn(64, hidden_size), indexer(64))
train_predictions = output.train
prior_predictions = output.prior

# %%
train_predictions.shape

# %%
indexer(0)

# %% [markdown]
# For training, we'll follow the same principle as is the paper: simply masking the unknown dimensions when calculating losses. This is hopefully sufficiently in keeping with the nature of SGD. Future work might explore using uncertainty to more cleverly compensate for missing values.

# %%
#|export
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
def train_epinet(epinet, indexer, embeddings, sparse_rewards, hidden_size,
                 batch_size=64, num_epochs=100, lr=1e-3):
    """
    Train the epinet using masked MSE loss.

    Args:
        epinet: The epinet MLP model
        indexer: The indexer function for epinet
        embeddings: Input embeddings tensor
        sparse_rewards: Sparse reward labels tensor
        hidden_size: Size of the hidden dimension
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epinet = epinet.to(device)

    # Setup optimizer
    optimizer = optim.Adam(epinet.parameters(), lr=lr)

    # Calculate number of batches
    n_samples = embeddings.shape[0]
    n_batches = n_samples // batch_size

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Shuffle data
        perm = torch.randperm(n_samples)
        embeddings = embeddings[perm]
        sparse_rewards = sparse_rewards[perm]

        # Batch training
        pbar = tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{num_epochs}')
        for b in pbar:
            # Get batch and move to device
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            batch_embeddings = embeddings[start_idx:end_idx].to(device)
            batch_rewards = sparse_rewards[start_idx:end_idx].to(device)

            # Generate random indices for epinet
            indices = indexer(batch_size).to(device)

            # Forward pass
            epiout = epinet(batch_embeddings, indices)
            predicted_rewards = epiout.train + epiout.prior # preweighted sum of the learnable and fixed components 
            
            # Create mask for non-nan values
            mask = ~torch.isnan(batch_rewards)

            # Calculate masked MSE loss
            loss = torch.mean((predicted_rewards[mask] - batch_rewards[mask])**2)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss/(b+1)})

            # Free memory
            del batch_embeddings
            del batch_rewards
            torch.cuda.empty_cache()

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss/n_batches:.6f}')

    return epinet


# %% [markdown]
# Now we train! From above, Ridge regression without the epistemic nn achieved a loss of 0.022. Let's see if we can match that in the same order of magnitude, and perhaps even get below it.
# Although as we've also changed the methodology (to gradient descent), and are adding randomness to the outputs, the raw numbers aren't directly comparable.

# %% [markdown]
# As seen below, we can quickly get in the same order of magnitude; the remainder of training is how to exploit randomness efficiently to further minimize the loss. It will be a good sanity check to see how much randomness is added: is the plain MLP doing most of the work, or are outputs of a substantial magnitude coming from the randomized portion of the network?

# %%
trained_epinet = train_epinet(
    epinet=epinet,
    indexer=indexer,
    embeddings=embeddings,
    sparse_rewards=sparse_rewards,
    hidden_size=hidden_size,
    lr = 1e-5
)

# %% [markdown]
# # Results

# %%

# %% [markdown]
# # Conclusion

# %%
