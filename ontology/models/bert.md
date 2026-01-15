---
entity_key: bert
title: BERT (Bidirectional Encoder Representations from Transformers)
parent: transformer
children: []
related:
  - gpt
  - llama
  - attention
  - supervised-learning
tags: [model, language-model, encoder]
---

# BERT (Bidirectional Encoder Representations from Transformers)

BERT is an encoder-only [[transformer|Transformer]] model developed by Google (2018). It introduced bidirectional pre-training for language understanding.

## Architecture

BERT uses the encoder portion of the [[transformer|Transformer]]:

- **Encoder-Only**: No decoder, just stacked encoder blocks
- **Bidirectional [[attention|Attention]]**: Each token attends to all tokens
- **[CLS] Token**: Special token for classification tasks
- **[SEP] Token**: Separates sentence pairs

### Model Sizes

| Variant | Layers | Hidden | Heads | Parameters |
|---------|--------|--------|-------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

## Training Objectives

BERT uses two [[unsupervised-learning|unsupervised]] objectives:

### Masked Language Modeling (MLM)

Randomly mask 15% of tokens, predict original:
```
Input:  The cat [MASK] on the mat
Target: sat
```

This enables bidirectional context.

### Next Sentence Prediction (NSP)

Predict if two sentences are consecutive:
```
[CLS] The cat sat [SEP] It was happy [SEP] → IsNext
[CLS] The cat sat [SEP] Pizza is good [SEP] → NotNext
```

## Key Innovation: Bidirectionality

Unlike [[gpt|GPT]] which only sees left context:
- GPT: "The cat ___" (predict next)
- BERT: "The ___ sat" (see both sides)

This captures richer representations for understanding tasks.

## Fine-Tuning

BERT excels at [[supervised-learning|supervised]] downstream tasks:

1. **Classification**: Add classifier on [CLS] token
2. **Named Entity Recognition**: Classify each token
3. **Question Answering**: Predict answer span
4. **Semantic Similarity**: Compare [CLS] embeddings

## Comparison with GPT

| Aspect | BERT | [[gpt|GPT]] |
|--------|------|-----|
| Pre-training | MLM + NSP | Next token |
| Direction | Bidirectional | Left-to-right |
| Best For | Understanding | Generation |
| Fine-tuning | Required | Optional |

## Variants

- **RoBERTa**: Better training, no NSP
- **ALBERT**: Parameter sharing, smaller
- **DistilBERT**: Knowledge distillation, 6 layers
- **ELECTRA**: Replaced token detection

## Impact on NLP

BERT revolutionized:
- Search engines (Google uses BERT)
- Sentiment analysis
- Question answering systems
- [[machine-learning|ML]] for text classification

## Training

Uses [[backpropagation|backpropagation]] with:
- 3.3B words (BooksCorpus + Wikipedia)
- 40 epochs for base, 40 for large
- [[deep-learning|Deep]] [[neural-network|networks]]
