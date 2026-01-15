---
entity_key: deep-learning
title: Deep Learning
parent: machine-learning
children:
  - neural-network
  - transformer
  - attention
related:
  - transformer
  - backpropagation
  - attention
tags: [concept, neural-networks, ai]
---

# Deep Learning

Deep learning is a subset of [[machine-learning|machine learning]] that uses [[neural-network|neural networks]] with multiple layers (hence "deep") to learn hierarchical representations of data.

## Architecture

Deep learning models consist of:

1. **Input Layer**: Receives raw data (images, text, audio)
2. **Hidden Layers**: Multiple layers that transform representations
3. **Output Layer**: Produces predictions or embeddings

Training uses [[backpropagation|backpropagation]] to compute gradients and update weights.

## Key Innovations

### Attention Mechanism

The [[attention|attention mechanism]] allows models to focus on relevant parts of the input. This is foundational to [[transformer|Transformer]] architectures.

### Transformer Architecture

[[transformer|Transformers]] replaced recurrent networks for sequence modeling. They enable parallel processing and better long-range dependencies.

## Model Families

### Language Models

- [[gpt|GPT]] - Generative Pre-trained Transformer for text generation
- [[bert|BERT]] - Bidirectional Encoder for language understanding
- [[llama|LLaMA]] - Open-source language model family

### Vision Models

- CNNs for image classification
- Vision Transformers (ViT)
- Diffusion models for image generation

## Training at Scale

Modern deep learning requires:
- Large datasets (billions of examples)
- Massive compute (thousands of GPUs)
- Efficient [[backpropagation|optimization algorithms]]

## Why "Deep"?

The depth (number of layers) enables learning hierarchical features:
- Early layers: edges, textures
- Middle layers: parts, patterns
- Later layers: objects, concepts
