---
entity_key: supervised-learning
title: Supervised Learning
parent: machine-learning
children: []
related:
  - unsupervised-learning
  - neural-network
  - backpropagation
tags: [concept, training, paradigm]
---

# Supervised Learning

Supervised learning is a [[machine-learning|machine learning]] paradigm where models learn from labeled examples. Each training sample includes both input features and the correct output (label).

## How It Works

1. **Training Data**: Pairs of (input, label)
2. **Model**: Learns mapping f(input) → label
3. **Loss Function**: Measures prediction errors
4. **Optimization**: [[backpropagation|Backpropagation]] minimizes loss

## Task Types

### Classification

Predicting discrete categories:
- Binary: spam/not spam
- Multi-class: cat/dog/bird
- Multi-label: multiple tags per item

### Regression

Predicting continuous values:
- House prices
- Temperature forecasting
- Stock predictions

## Common Algorithms

### Traditional ML

- Linear/Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines

### Deep Learning

- [[neural-network|Neural Networks]]
- [[transformer|Transformers]] for sequences
- [[bert|BERT]] for text classification

## Training Process

```
for each epoch:
    for each batch:
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)
        gradients = backprop(loss)
        update_weights(gradients)
```

## Evaluation

- **Train/Test Split**: Evaluate on unseen data
- **Cross-Validation**: Multiple train/test splits
- **Metrics**: Accuracy, F1, RMSE, etc.

## Contrast with Unsupervised

Unlike [[unsupervised-learning|unsupervised learning]], supervised learning requires labeled data. Labels are often expensive to obtain, which limits dataset sizes.

## Modern Applications

- [[gpt|GPT]] trained on next-token prediction (self-supervised)
- [[bert|BERT]] fine-tuned on labeled downstream tasks
- Image classification with [[neural-network|CNNs]]
