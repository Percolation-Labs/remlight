---
entity_key: backpropagation
title: Backpropagation
parent: machine-learning
children: []
related:
  - neural-network
  - deep-learning
  - supervised-learning
tags: [technique, training, optimization]
---

# Backpropagation

Backpropagation (backprop) is the algorithm used to train [[neural-network|neural networks]]. It efficiently computes gradients of the loss function with respect to weights.

## How It Works

### 1. Forward Pass

Compute predictions by propagating input through the network:

```
x → Layer1 → a1 → Layer2 → a2 → ... → Output → Loss
```

### 2. Backward Pass

Compute gradients from output to input using chain rule:

```
∂L/∂w = ∂L/∂output × ∂output/∂hidden × ∂hidden/∂w
```

### 3. Weight Update

Apply gradients to update weights:

```
w_new = w_old - learning_rate × ∂L/∂w
```

## The Chain Rule

Backpropagation is just the chain rule applied systematically:

```
∂L/∂w_1 = ∂L/∂a_n × ∂a_n/∂a_{n-1} × ... × ∂a_2/∂a_1 × ∂a_1/∂w_1
```

Each layer computes its local gradient, and these are chained together.

## Computational Graph

Modern [[deep-learning|deep learning]] frameworks (PyTorch, TensorFlow) build computational graphs:

1. **Forward**: Build graph, compute outputs
2. **Backward**: Traverse graph in reverse, compute gradients
3. **Automatic Differentiation**: Framework handles chain rule

## Optimizers

Backpropagation computes gradients. Optimizers decide how to use them:

### Stochastic Gradient Descent (SGD)

```
w = w - lr × gradient
```

### Momentum

```
v = momentum × v - lr × gradient
w = w + v
```

### Adam

Combines momentum with adaptive learning rates. Most common for [[transformer|Transformers]].

## Challenges

### Vanishing Gradients

In deep networks, gradients can shrink exponentially:
- Solutions: ReLU activation, residual connections, LayerNorm
- [[transformer|Transformers]] use these extensively

### Exploding Gradients

Gradients can grow exponentially:
- Solutions: Gradient clipping, careful initialization

### Computational Cost

For [[deep-learning|deep]] models with billions of parameters:
- Memory: Store activations for backward pass
- Compute: Same FLOPs as forward pass
- Solutions: Gradient checkpointing, mixed precision

## Training Loop

Complete training iteration:

```python
for batch in data:
    # Forward
    predictions = model(batch.inputs)
    loss = loss_fn(predictions, batch.targets)

    # Backward
    loss.backward()  # Computes all gradients

    # Update
    optimizer.step()  # Applies gradients
    optimizer.zero_grad()  # Reset for next batch
```

## Historical Note

Backpropagation was popularized by Rumelhart, Hinton, and Williams (1986), enabling practical training of [[neural-network|neural networks]] and later [[deep-learning|deep learning]].

## Applications

Used to train all modern neural networks:
- [[gpt|GPT]] language models
- [[bert|BERT]] encoders
- [[transformer|Transformers]] of all kinds
- [[machine-learning|ML]] systems with learnable parameters
