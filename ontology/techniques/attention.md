---
entity_key: attention
title: Attention Mechanism
parent: deep-learning
children: []
related:
  - transformer
  - neural-network
  - gpt
  - bert
tags: [technique, architecture, core]
---

# Attention Mechanism

The attention mechanism allows [[neural-network|neural networks]] to focus on relevant parts of the input when producing an output. It's the core innovation behind [[transformer|Transformers]].

## Core Concept

Attention computes a weighted sum of values, where weights depend on query-key similarity:

```
Attention(Q, K, V) = softmax(score(Q, K)) × V
```

This allows the model to "attend to" (focus on) relevant information.

## Scaled Dot-Product Attention

The standard attention used in [[transformer|Transformers]]:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q (Query)**: What we're looking for (d_k dimensions)
- **K (Key)**: What we're matching against (d_k dimensions)
- **V (Value)**: What we retrieve (d_v dimensions)
- **√d_k**: Scaling factor for stable gradients

## Multi-Head Attention

Run attention multiple times in parallel:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Each head can learn different attention patterns:
- One head: syntactic relationships
- Another head: semantic similarity
- Another head: position-based patterns

## Self-Attention

When Q, K, V all come from the same sequence:

```
Self-Attention(X) = Attention(XW^Q, XW^K, XW^V)
```

Every position attends to every other position. This captures relationships within a sequence.

## Types of Attention

### Causal (Autoregressive)

Used in [[gpt|GPT]] - each position only attends to previous positions:

```
mask = tril(ones(n, n))  # Lower triangular
scores = QK^T / √d_k
scores = scores.masked_fill(mask == 0, -inf)
```

### Bidirectional

Used in [[bert|BERT]] - each position attends to all positions. Better for understanding but can't generate.

### Cross-Attention

Query from one sequence, key/value from another:
- Decoder attending to encoder (original [[transformer|Transformer]])
- Multimodal: text attending to image

## Computational Complexity

Standard attention is O(n²) in sequence length:
- Memory: Store n × n attention matrix
- Compute: n² dot products

This limits context length. Solutions:
- Sparse attention
- Linear attention
- Flash Attention (optimized CUDA kernels)

## Attention Patterns

Trained models learn meaningful patterns:
- Syntactic heads: subject-verb agreement
- Positional heads: attend to nearby tokens
- Semantic heads: attend to related concepts

## Why Attention Works

1. **Direct connections**: Any position can attend to any other
2. **Flexible weighting**: Learned, content-based weights
3. **Parallel computation**: All positions computed simultaneously
4. **Interpretable**: Attention weights show model focus

## Impact

Attention revolutionized [[deep-learning|deep learning]]:
- Replaced recurrence for sequences
- Enabled [[gpt|GPT]], [[bert|BERT]], [[llama|LLaMA]]
- Extended to vision, audio, multimodal
- Core of all modern [[machine-learning|ML]] systems

## Training

Attention weights are learned via [[backpropagation|backpropagation]]:
- Gradients flow through softmax
- Q, K, V projection weights updated
- Head weights (W_O) learned
