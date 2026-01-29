# Unsupervised Autoencoder on MNIST

## Overview

This repository contains an implementation of a **deep autoencoder trained in an unsupervised manner** on the MNIST dataset using PyTorch. The objective is to learn compact latent representations of handwritten digits through reconstruction, without relying on labels during training.

The implementation is inspired by pedagogical and research-oriented work on representation learning, including approaches similar to those presented by **Louis Bagot**.

---

## Notebook Structure and Chunk-by-Chunk Explanation

### Chunk 1 – Imports and dependencies

This block imports all required libraries:

* `torch`, `torch.nn`, `torch.nn.functional`: core PyTorch components for model definition and training.
* `numpy`: numerical utilities.
* `matplotlib`: visualization of images and reconstructions.
* `torchvision.datasets` and `ToTensor`: loading and preprocessing the MNIST dataset.

This chunk sets up the computational and visualization environment.

---

### Chunk 2 – Dataset loading (MNIST)

The MNIST dataset is downloaded and loaded:

* Training set (`train=True`) used for unsupervised learning.
* Test set (`train=False`) used for evaluation and visualization.

Images are converted to tensors and normalized to `[0, 1]` via `ToTensor()`.

---

### Chunk 3 – Data visualization

A 5×5 grid of MNIST images is displayed:

* Allows sanity-checking of the dataset.
* Shows the correspondence between raw images and their ground-truth labels (labels are **not** used during training).

This step is purely exploratory.

---

### Chunk 4 – DataLoaders

PyTorch `DataLoader` objects are created for:

* Training data (shuffled, batched).
* Test data.

This enables efficient mini-batch training and evaluation.

---

### Chunk 5 – Batch shape inspection

A single batch is extracted from the test loader to print:

* Image tensor shape.
* Label tensor shape.

This confirms correct batching and tensor dimensions before model definition.

---

### Chunk 6 – Demonstration hyperparameters

This block defines example hyperparameters:

* `HIDDEN_WIDTH`: size of the latent representation.
* `ACTIVATION`: activation function.

A comment explicitly notes that these values are **for demonstration only** and should be modified later in the dedicated hyperparameter section.

---

### Chunk 7 – Autoencoder architecture (`Net`)

Defines the neural network model:

**Encoder**:

* Fully connected layers reducing dimensionality from 784 → 64.
* Non-linear activations introduce representational capacity.

**Decoder**:

* Symmetric fully connected layers expanding 64 → 784.
* Final output reconstructs the input image.

This architecture implements a classic deep autoencoder for dimensionality reduction.

---

### Chunk 8 – Forward pass sanity check

A single MNIST image is:

* Normalized.
* Passed through the autoencoder.

This verifies that the model produces an output with the expected shape.

---

### Chunk 9 – Model reset utility

Defines a helper function `reset` that:

* Instantiates a new model.
* Creates a loss function (criterion).
* Creates an optimizer with a specified learning rate.
-> The learning rate determines how quickly and with what stability your network learns by adjusting its weights to minimize loss.

This abstraction simplifies repeated experiments and hyperparameter tuning.

---

### Chunk 10 – Training loop

Defines the main training routine:

* Iterates over epochs and mini-batches.
* Performs forward pass, loss computation, backpropagation, and optimization.
* Tracks training and test losses, accuracy, and steps in a `logs` dictionary.

This function encapsulates the learning dynamics of the autoencoder.

---

### Chunk 11 – Test evaluation and logging

Defines a helper function to:

* Evaluate the model on the test set.
* Compute reconstruction loss and accuracy metrics.
* Log and print progress during training.

This enables monitoring of generalization and convergence.

---

### Chunk 12 – (Optional) Plotting utilities

Commented-out plotting code:

* Intended to visualize training and test losses.
* Uses logarithmic scale for better interpretability.

Kept for future experimentation and analysis.

---

### Chunk 13 – Hyperparameter configuration

Centralized hyperparameter box:

* Latent dimension size.
* Activation function.
* Learning rate.
* Optimizer.
* Loss function (`MSELoss`, standard for autoencoders).

This is the recommended location for launching new experimental runs.

---

### Chunk 14 – Model training execution

Initializes:

* Network.
* Loss function.
* Optimizer.

Launches training for a fixed number of epochs and stores logs.

---

### Chunk 15 – Reconstruction visualization

Selects a specific digit ("2") from the test set:

* Runs it through the trained autoencoder.
* Reshapes the output into a 28×28 image.
* Compares the original image with its reconstruction.

This qualitative evaluation illustrates the effectiveness of the learned latent representation.

---

## Objectives

* Learn compact latent representations from unlabeled data.
* Demonstrate unsupervised learning with deep autoencoders.
* Provide a clear, modular, and extensible PyTorch implementation.

## Possible Extensions

* Visualization of latent space (PCA / t-SNE).
* Denoising autoencoder.
* Variational autoencoder (VAE).
* Experiments with different latent dimensions and activation functions.

## References

* Hinton & Salakhutdinov, *Reducing the Dimensionality of Data with Neural Networks*.
* Pedagogical and research work on representation learning, including contributions by **Louis Bagot**.
# Implementation_of_unsupervised_Autoencoder_using_deep_learning