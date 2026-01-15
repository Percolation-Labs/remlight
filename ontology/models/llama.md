---
entity_key: llama
title: LLaMA (Large Language Model Meta AI)
parent: transformer
children: []
related:
  - gpt
  - bert
  - attention
  - deep-learning
tags: [model, language-model, open-source]
---

# LLaMA (Large Language Model Meta AI)

LLaMA is a family of open-weight [[transformer|Transformer]] models developed by Meta. It democratized access to large language models for research.

## Architecture

LLaMA uses a decoder-only [[transformer|Transformer]] similar to [[gpt|GPT]]:

- **Pre-normalization**: RMSNorm before attention (not after)
- **SwiGLU Activation**: Replaces ReLU in feed-forward
- **Rotary Position Embeddings (RoPE)**: Better position encoding
- **No Bias Terms**: Removed from most linear layers

### Model Sizes

| Version | Sizes Available |
|---------|-----------------|
| LLaMA 1 | 7B, 13B, 33B, 65B |
| LLaMA 2 | 7B, 13B, 70B |
| LLaMA 3 | 8B, 70B, 405B |

## Key Innovations

### Efficient Training

LLaMA showed smaller models trained on more data outperform larger models:
- 7B model competitive with GPT-3 (175B)
- Trained on 1-1.4 trillion tokens
- Efficient [[backpropagation|training]] at scale

### Open Weights

Unlike [[gpt|GPT]], weights are publicly available:
- Enables research and fine-tuning
- Spawned ecosystem (Alpaca, Vicuna, etc.)
- Community can inspect and improve

### Rotary Position Embeddings

RoPE encodes position via rotation in complex space:
- Better than learned position embeddings
- Enables length extrapolation
- Used in [[attention|attention]] computation

## Training

LLaMA training uses:
- Publicly available data only
- [[unsupervised-learning|Unsupervised]] next-token prediction
- AdamW optimizer with cosine schedule
- [[deep-learning|Deep]] [[neural-network|networks]]

### Data Sources

- CommonCrawl (67%)
- C4 (15%)
- GitHub (4.5%)
- Wikipedia (4.5%)
- Books, ArXiv, StackExchange

## Comparison

| Aspect | LLaMA | [[gpt|GPT]] | [[bert|BERT]] |
|--------|-------|-----|------|
| Architecture | Decoder | Decoder | Encoder |
| Weights | Open | Closed | Open |
| Best For | Generation | Generation | Understanding |
| Context | 4K-128K | 8K-128K | 512 |

## Fine-Tuned Variants

Community-created models built on LLaMA:
- **Alpaca**: Instruction-tuned on 52K examples
- **Vicuna**: Fine-tuned on ChatGPT conversations
- **CodeLlama**: Specialized for code
- **Llama-2-Chat**: Official instruction-tuned

## Impact

LLaMA changed the landscape:
- Enabled open [[machine-learning|ML]] research
- Showed efficient training matters
- Created vibrant open-source ecosystem
- Alternative to proprietary models
