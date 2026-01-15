---
entity_key: transformer
title: Transformer
parent: deep-learning
children:
  - gpt
  - bert
  - llama
related:
  - attention
  - neural-network
  - backpropagation
tags: [model, architecture, nlp]
---

# Transformer

The Transformer is a [[neural-network|neural network]] architecture introduced in "Attention Is All You Need" (2017). It revolutionized NLP by replacing recurrence with [[attention|self-attention]].

## Architecture

### Encoder-Decoder Structure

Original Transformer has:
- **Encoder**: Processes input sequence → contextual representations
- **Decoder**: Generates output sequence autoregressively

### Key Components

1. **Multi-Head Self-Attention**: [[attention|Attention]] across all positions
2. **Feed-Forward Networks**: Position-wise transformations
3. **Layer Normalization**: Stabilizes training
4. **Residual Connections**: Enables deep networks

## Self-Attention Mechanism

The [[attention|attention mechanism]] computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q (Query): What we're looking for
- K (Key): What we match against
- V (Value): What we retrieve

Multi-head attention runs this in parallel with different projections.

## Positional Encoding

Since attention is permutation-invariant, position information is added:
- Sinusoidal encodings (original)
- Learned position embeddings ([[gpt|GPT]], [[bert|BERT]])
- Rotary position embeddings ([[llama|LLaMA]])

## Variants

### Encoder-Only

- [[bert|BERT]]: Bidirectional, masked language modeling
- Best for understanding tasks (classification, NER)

### Decoder-Only

- [[gpt|GPT]]: Autoregressive, next-token prediction
- Best for generation tasks

### Encoder-Decoder

- T5, BART: Sequence-to-sequence tasks
- Translation, summarization

## Why Transformers Dominate

1. **Parallelization**: Unlike RNNs, all positions computed simultaneously
2. **Long-Range Dependencies**: [[attention|Attention]] directly connects all positions
3. **Scalability**: Performance improves with model size and data
4. **Transfer Learning**: Pre-train once, fine-tune for many tasks

## Training

Transformers are trained with [[backpropagation|backpropagation]] on massive datasets:
- [[gpt|GPT]]: Trillions of tokens
- [[bert|BERT]]: Billions of tokens
- Requires significant compute (thousands of GPUs)

## Impact

Transformers power:
- Large Language Models (ChatGPT, Claude)
- [[machine-learning|Modern ML]] systems
- Vision Transformers (ViT)
- Multimodal models
