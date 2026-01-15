---
entity_key: gpt
title: GPT (Generative Pre-trained Transformer)
parent: transformer
children: []
related:
  - bert
  - llama
  - attention
  - deep-learning
tags: [model, language-model, generative]
---

# GPT (Generative Pre-trained Transformer)

GPT is a family of decoder-only [[transformer|Transformer]] models developed by OpenAI. It pioneered the paradigm of large-scale pre-training followed by fine-tuning.

## Architecture

GPT uses the decoder portion of the [[transformer|Transformer]]:

- **Decoder-Only**: No encoder, just stacked decoder blocks
- **Causal Attention**: Each token attends only to previous tokens
- **Autoregressive**: Generates one token at a time

### Model Sizes

| Version | Parameters | Training Data |
|---------|------------|---------------|
| GPT-1   | 117M       | BookCorpus    |
| GPT-2   | 1.5B       | WebText       |
| GPT-3   | 175B       | Common Crawl  |
| GPT-4   | Unknown    | Proprietary   |

## Training Objective

GPT uses next-token prediction:

```
L = -Σ log P(token_i | token_1, ..., token_{i-1})
```

This is [[unsupervised-learning|unsupervised]] - no labeled data needed, just text.

## Key Innovations

### GPT-1: Transfer Learning

Showed pre-training on unlabeled text + fine-tuning works for NLP.

### GPT-2: Zero-Shot Capabilities

Demonstrated language models can perform tasks without fine-tuning.

### GPT-3: In-Context Learning

Introduced few-shot prompting - examples in the prompt guide behavior.

### ChatGPT: RLHF

Added Reinforcement Learning from Human Feedback for instruction-following.

## Comparison with BERT

| Aspect | GPT | [[bert|BERT]] |
|--------|-----|------|
| Direction | Left-to-right | Bidirectional |
| Training | Next token | Masked tokens |
| Best For | Generation | Understanding |
| [[attention|Attention]] | Causal | Full |

## Emergent Abilities

Large GPT models show emergent capabilities:
- Chain-of-thought reasoning
- Code generation
- Mathematical problem solving
- Multilingual translation

These abilities appear suddenly as models scale.

## Usage

GPT models power:
- ChatGPT, GPT-4 API
- Code assistants (Copilot)
- Content generation
- Question answering

## Training Details

- [[backpropagation|Backpropagation]] on massive corpora
- [[deep-learning|Deep]] [[neural-network|networks]] with billions of parameters
- Months of training on thousands of GPUs
- Learned [[attention|attention]] patterns encode knowledge
