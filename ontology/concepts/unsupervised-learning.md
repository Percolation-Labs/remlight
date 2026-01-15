---
entity_key: unsupervised-learning
title: Unsupervised Learning
parent: machine-learning
children: []
related:
  - supervised-learning
  - deep-learning
  - attention
tags: [concept, training, paradigm]
---

# Unsupervised Learning

Unsupervised learning discovers patterns in data without labeled examples. The model finds structure on its own.

## How It Works

1. **Unlabeled Data**: Just inputs, no labels
2. **Pattern Discovery**: Model finds clusters, structures, representations
3. **No Explicit Objective**: Learns from data distribution itself

## Task Types

### Clustering

Grouping similar items:
- K-Means
- Hierarchical clustering
- DBSCAN

### Dimensionality Reduction

Compressing high-dimensional data:
- PCA (Principal Component Analysis)
- t-SNE for visualization
- Autoencoders

### Representation Learning

Learning useful features for downstream tasks:
- Word embeddings (Word2Vec)
- [[bert|BERT]] pre-training
- Contrastive learning

### Generative Modeling

Learning to generate new data:
- [[gpt|GPT]] learns language distribution
- VAEs for image generation
- Diffusion models

## Self-Supervised Learning

A powerful variant where supervision comes from the data itself:

- **Masked Language Modeling**: Predict masked words ([[bert|BERT]])
- **Next Token Prediction**: Predict next word ([[gpt|GPT]])
- **Contrastive Learning**: Match augmented views

This bridges unsupervised and [[supervised-learning|supervised learning]].

## Applications

- **Pre-training**: [[deep-learning|Deep learning]] models trained on unlabeled data
- **Embeddings**: Dense vector representations for similarity
- **Anomaly Detection**: Find unusual patterns
- **Data Exploration**: Understand dataset structure

## Advantages

- Abundant unlabeled data available
- Learns general representations
- Transfers to multiple tasks

## Relationship to Other Paradigms

Contrasts with [[supervised-learning|supervised learning]] which requires labels. Modern [[transformer|Transformers]] often use unsupervised pre-training followed by supervised fine-tuning.
