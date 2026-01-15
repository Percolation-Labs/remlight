---
entity_key: neural-network
title: Neural Network
parent: deep-learning
children: []
related:
  - backpropagation
  - machine-learning
  - attention
tags: [concept, architecture, foundation]
---

# Neural Network

A neural network is a computational model inspired by biological neurons. It consists of interconnected nodes (neurons) organized in layers that process information.

## Structure

### Neurons

Each neuron:
1. Receives inputs from previous layer
2. Computes weighted sum: `z = Σ(w_i * x_i) + b`
3. Applies activation function: `a = f(z)`
4. Passes output to next layer

### Layers

- **Input Layer**: Receives raw features
- **Hidden Layers**: Learn intermediate representations
- **Output Layer**: Produces final predictions

## Activation Functions

Non-linear functions that enable learning complex patterns:

- **ReLU**: `max(0, x)` - most common in hidden layers
- **Sigmoid**: `1 / (1 + e^(-x))` - for binary classification
- **Softmax**: Normalizes outputs to probabilities
- **Tanh**: `(e^x - e^(-x)) / (e^x + e^(-x))`

## Training

Neural networks learn via [[backpropagation|backpropagation]]:

1. **Forward Pass**: Compute predictions
2. **Loss Calculation**: Measure error
3. **Backward Pass**: Compute gradients
4. **Update Weights**: Apply optimizer (SGD, Adam)

## Types of Networks

### Feedforward Networks

Simple networks where information flows one direction. Used in [[machine-learning|basic ML]] tasks.

### Convolutional Networks (CNNs)

Specialized for spatial data (images). Use convolution operations to detect local patterns.

### Recurrent Networks (RNNs)

Process sequential data with memory. Largely replaced by [[transformer|Transformers]] for NLP.

### Transformers

Modern architecture using [[attention|attention mechanisms]]. Powers [[gpt|GPT]], [[bert|BERT]], and other large models.

## Universal Approximation

Neural networks can approximate any continuous function given sufficient neurons - this is the Universal Approximation Theorem.

## Relationship to Deep Learning

When neural networks have many layers, we call it [[deep-learning|deep learning]]. Depth enables learning hierarchical representations.
