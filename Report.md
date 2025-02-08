# Next Word Prediction Using Language Models

## 1. Introduction
Next Word Prediction (NWP) involves predicting the most probable word following a given sequence of words. Mathematically, given a sequence of words $w_1, w_2, ..., w_{n-1}$, the goal is to determine the next word $w_n$ such that:

$$
w_n = \arg\max_{w \in V} P(w | w_1, w_2, ..., w_{n-1})
$$

where $V$ is the vocabulary and $P(w | w_1, w_2, ..., w_{n-1})$ is the conditional probability of $w$ given the preceding words.

---

## 2. Language Models Implemented

### 2.1 Feed Forward Neural Network (FFNN) Language Model
A Feed Forward Neural Network (FFNN) model predicts the next word by processing a fixed-size window of preceding words using fully connected layers with activation functions. The model outputs probabilities for all words in the vocabulary, and predictions are based on the top $k$ probabilities.

- **Implementation Details:**
  - Used **n-grams** with $n = 3$ and $n = 5$.
  - Input is passed through an embedding layer, followed by fully connected layers.
  - The final layer outputs word probabilities using the softmax function.

---

### 2.2 Vanilla Recurrent Neural Network (RNN) Language Model
A Vanilla Recurrent Neural Network (RNN) maintains a hidden state to capture sequential dependencies of preceding words. It iteratively processes input word embeddings and outputs probabilities for the next word.

- **Implementation Details:**
  - Embedding layer followed by an RNN layer.
  - Final layer outputs probabilities for each word in the vocabulary.

---

### 2.3 Long Short-Term Memory (LSTM) Language Model
An LSTM model extends RNNs by introducing memory cells and gating mechanisms (input, forget, and output gates) to capture long-term dependencies in sequences. It effectively mitigates the vanishing gradient problem.

- **Implementation Details:**
  - Similar architecture to RNN but with LSTM layers instead.
  - Uses gating mechanisms to manage long-range dependencies.

---

## 3. Corpus Used
The models were trained on the following corpora:

- **Pride and Prejudice corpus** (124,970 words)
- **Ulysses corpus** (268,117 words)

Each dataset was preprocessed and tokenized before training.

---

## 4. Perplexity Analysis

### 4.1 Perplexity Calculation
Perplexity is used to evaluate model performance, defined as:

$$
PPL = e^{-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, ..., w_{i-1})}
$$

where $N$ is the number of words in the sequence.

**Steps:**
1. Compute perplexity scores for all sentences in both corpora.
2. Calculate the **average perplexity** for each model on the training set.
3. Evaluate and report **test set perplexities**.

---

### 4.2 Reporting Perplexity Scores

In the directory ```Text_Files``` you can find .txt files in the following format:

For every model $\{lm\_type\}\_\{set\}\_\{n\}\_\{corpus\}.txt$

where,

lm_type $\in$ \{'ffnn', 'rnn', 'lstm'\}
<br>
set $\in$ \{'train', 'test\}
<br>
n $\in$ \{3,5\}
<br>
corpus $\in$ \{'Pride and Prejudice', 'Ulysses'\}


---

### 4.3 Comparison and Ranking

* As compared to Assignment 1, the models here perform better in terms of perplexity scores of train and test sets.

* Ranking of models:

1. LSTM

2. RNN

3. FFNN

---

## 5. Observations and Insights

1. **Model Performance for Longer Sentences:**
   - **LSTM performed best** due to its ability to capture long-range dependencies.
   - **RNN struggled** with longer sentences due to vanishing gradient issues.
   - **FFNN had limited context awareness** as it only considered a fixed-size window.

2. **Effect of N-gram Size on FFNN Model:**
   - **Higher n-grams (n=5) improved performance** slightly by capturing more context.
   - **Lower n-grams (n=3) resulted in lower performance** due to insufficient context.
   - However, **increasing n too much leads to sparsity issues**.

---

## 6. Conclusion
- **LSTM outperforms both FFNN and RNN** in next-word prediction due to its ability to retain long-term dependencies.
- **FFNN is constrained by fixed window size**, making it less effective for longer contexts.
- **RNN performs better than FFNN but struggles with long sequences** due to vanishing gradients.
- **Using a larger n-gram improves FFNN but has diminishing returns beyond a certain point**.