# Sequential-Network-Architecture

# Cyberbullying Tweet Classification with Sequential Neural Architectures

## Overview

This project aims to classify tweets as cyberbullying or non-cyberbullying using various **sequential network architectures**, including **Convolutional Neural Networks (CNNs)** and **Transformer-based models**. The dataset is a balanced binary classification problem sourced from Kaggle.

## Dataset

**Source:** [Cyberbullying Tweets Dataset on Kaggle](https://www.kaggle.com/datasets/soorajtomar/cyberbullying-tweets/data)

**Structure:**
- `Text`: The tweet (string)
- `CB_Label`: Binary label (0 = Not cyberbullying, 1 = Cyberbullying)

**Characteristics:**
- Total tweets: 11,101
- Balanced labels: 50% positive, 50% negative
- Final dataset (after preprocessing): 10,757 tweets

## Data Preparation

- **Preprocessing Techniques:**
  - Tokenization of text into integer sequences
  - Tweets truncated/padded to a sequence length of **35**
  - Tweets longer than 50 characters removed
  - Labels left as integer (no one-hot encoding)

- **Final Dataset:**
  - Feature matrix shape: `(10757, 35)`
  - Label vector shape: `(10757,)`
  - Class distribution: `[5537, 5220]`

## Evaluation Metrics

- **Primary Metric:** `F1 Score`
  - Balances precision and recall
  - Appropriate for minimizing false positives (mislabeling innocents) and false negatives (missing real cyberbullying)
- Accuracy and validation loss also tracked for comparison
- **Statistical significance** assessed using **ANOVA**

## Train/Test Split

- **Method:** Random split with `stratified sampling`
- Maintains class balance in training and test sets
- Mirrors real-world deployment where new tweets appear randomly

## Techniques & Technologies Used

- **Libraries/Frameworks:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib
- **Embeddings:**
  - [GloVe](https://nlp.stanford.edu/projects/glove/)
  - [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch)
- **Regularization & Optimization:**
  - Dropout layers
  - L2 regularization
  - RMSprop optimizer
  - Early stopping
  - Learning rate scheduling

## Models and Architectures

### CNN 1
- **Layers:** 2 × Conv1D (128 filters, size 5), ReLU, MaxPooling, Dense
- **Embedding:** GloVe (100D)
- **Regularization:** Dropout (0.3, 0.4, 0.5)
- **F1 Score:** Train = `0.8640`, Test = `0.6769`
- **Observation:** Overfitting detected

### CNN 2
- **Changes:** Fewer filters (64), L2 regularization, higher dropout (0.5), reduced dense units
- **Results:** Train = `0.5123`, Test = `0.4895`
- **Observation:** Worse performance, possibly underfitting

### Transformer 1
- **Architecture:** 1 transformer block (2 heads, FFN = 32), dropout, global average pooling
- **Embedding:** GloVe (100D)
- **F1 Score:** Train = `0.7262`, Test = `0.7232`
- **Observation:** Some early overfitting but generalizes well

### Transformer 2
- **Changes:** 4 attention heads, FFN = 64, L2 regularization, dropout = 0.5, lower LR
- **F1 Score:** Train = `0.7203`, Test = `0.7180`
- **Observation:** Improved accuracy and generalization

### Transformer 3
- **Architecture:** Added second multi-headed self-attention layer
- **F1 Score:** Train = `0.7558`, Test = `0.7220`
- **Validation Accuracy:** `0.7416`
- **Validation Loss:** `0.5254`
- **Observation:** Best model overall

## Final Results Summary

| Model         | Validation Acc | Validation Loss | Train F1 | Test F1 |
|---------------|----------------|------------------|----------|---------|
| CNN1          | 0.7314         | 0.5464           | 0.8640   | 0.6769  |
| CNN2          | 0.6608         | 0.6847           | 0.5123   | 0.4895  |
| Transformer1  | 0.7119         | 0.5617           | 0.7262   | 0.7232  |
| Transformer2  | 0.7314         | 0.5464           | 0.7203   | 0.7180  |
| **Transformer3** | **0.7416**     | **0.5254**       | **0.7558** | **0.7220** |

## Embedding Comparison

| Embedding     | Validation Acc | Validation Loss | F1 Score |
|---------------|----------------|------------------|----------|
| GloVe         | 0.7416         | 0.5254           | 0.7220   |
| Numberbatch   | **0.7537**     | **0.5156**       | **0.7384** |

**Conclusion:** ConceptNet Numberbatch slightly outperforms GloVe in all metrics.

## Statistical Analysis

- **ANOVA Results:**
  - Accuracy: `F = 15.44`, `p = 0.0000` → Statistically significant
  - Loss: `F = 5.69`, `p = 0.0005` → Statistically significant

## Conclusion

- The **Transformer3 model with ConceptNet embeddings** provides the best performance across all metrics.
- **CNN1** performed decently but suffered from overfitting.
- **CNN2** underperformed despite regularization.
- **Transformer architectures** consistently outperformed CNNs.
- This project demonstrates the potential for **Transformer-based solutions in cyberbullying detection**, particularly when paired with semantically rich embeddings like **Numberbatch**.

---

