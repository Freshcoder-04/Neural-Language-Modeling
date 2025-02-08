import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from tokenizer import custom_nlp_tokenizer
import sys
import os
import pickle
import torch.nn.functional as F

class FFNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_gram):
        super(FFNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(n_gram * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take last output in the sequence
        return out


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take last output in the sequence
        return out


class TextDataset(Dataset):
    def __init__(self, ngrams, vocab):
        self.ngrams = ngrams
        self.vocab = vocab

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, idx):
        context, target = self.ngrams[idx]
        context_tensor = torch.tensor([self.vocab.get(word, 0) for word in context], dtype=torch.long)
        target_tensor = torch.tensor(self.vocab.get(target, 0), dtype=torch.long)
        return context_tensor, target_tensor


def generate_ngrams(tokenized_text, n):
    ngrams = []
    for sentence in tokenized_text:
        if len(sentence) < n:
            continue
        for i in range(len(sentence) - n):
            context = sentence[i:i + n]
            target = sentence[i + n]
            ngrams.append((context, target))
    return ngrams


class Training:
    def __init__(self, model, dataloader, vocab, inv_vocab, epochs=10, lr=0.001, save_path="/kaggle/working/"):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)
        # self.model = model.to(self.device)
        self.dataloader = dataloader
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.epochs = epochs
        self.lr = lr
        self.save_path = save_path

    def train_model(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            total_loss = 0
            for context, target in self.dataloader:
                # context, target = context.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(context)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(self.dataloader)}")

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path,'model.pt'))
        print(f"Full model saved at {self.save_path}")


def compute_perplexity2(model, dataloader):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    total_log_prob = 0
    total_words = 0

    with torch.no_grad():
        for context, target in dataloader:
            # context, target = context.to(device), target.to(device)
            output = model(context)
            log_probs = torch.log_softmax(output, dim=1)
            batch_log_prob = log_probs[torch.arange(target.shape[0]), target]
            total_log_prob += batch_log_prob.sum().item()
            total_words += target.shape[0]

    return torch.exp(torch.tensor(-total_log_prob / total_words))



def create_vocab(tokenized_text, path):
    vocab = {word: idx for idx, word in enumerate(set(sum(tokenized_text, [])))}
    inv_vocab = {idx: word for word, idx in vocab.items()}

    os.makedirs(path, exist_ok=True)

    vocab_path = os.path.join(path, "vocab.pkl")
    inv_vocab_path = os.path.join(path, "inv_vocab.pkl")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    with open(inv_vocab_path, "wb") as f:
        pickle.dump(inv_vocab, f)

    print(f"Vocabulary saved at {vocab_path} and {inv_vocab_path}")
    return vocab, inv_vocab


def split_dataset(ngram_data, test_size=1000):
    test_set = random.sample(ngram_data, test_size)
    train_set = [pair for pair in ngram_data if pair not in test_set]
    return train_set, test_set


def predict_next_word(model, vocab, inv_vocab, n_gram, sentence, k):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    
    if len(sentence) < n_gram:
        print("Sentence too short!")
        return

    context = sentence[-n_gram:]
    context_tensor = torch.tensor([vocab.get(word, 0) for word in context], dtype=torch.long).unsqueeze(0)
    output = model(context_tensor)
    probs = torch.softmax(output, dim=1).squeeze()
    top_k = torch.topk(probs, k)

    for idx, prob in zip(top_k.indices, top_k.values):
        print(f"{inv_vocab[idx.item()]} {prob.item():.4f}")

def compute_and_save_perplexities(model, train_loader, test_loader, vocab, inv_vocab, model_name, n_gram, corpus_name):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.eval()  # Set model to evaluation mode
    
    def compute_perplexity(loader, filename):
        results_dir = "/kaggle/working/results"
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, filename)
        print(filename)

        total_log_prob = 0
        total_words = 0
        perplexity_lines = []

        with torch.no_grad():
            for context, target in loader:
                # context, target = context.to(device), target.to(device)
                output = model(context)
                log_probs = F.log_softmax(output, dim=1)
                batch_log_prob = log_probs[torch.arange(target.shape[0]), target]
                
                total_log_prob += batch_log_prob.sum().item()
                total_words += target.shape[0]

                # Convert context indices back to words
                context_words = [" ".join(inv_vocab[idx.item()] for idx in sent) for sent in context]
                target_words = [inv_vocab[target[i].item()] for i in range(target.shape[0])]

                for ctx, tgt, ppl in zip(context_words, target_words, batch_log_prob):
                    perplexity_lines.append(f"{ctx} -> {tgt}\t{torch.exp(-ppl).item():.4f}\n")

        # Compute the average perplexity
        avg_perplexity = torch.exp(torch.tensor(-total_log_prob / total_words)).item()

        # Write to file with average perplexity at the top
        with open(path, "w") as f:
            f.write(f"Average Perplexity: {avg_perplexity:.4f}\n")
            f.writelines(perplexity_lines)

        print(f"Saved perplexity scores with sentences in: {path}")

    # Create filenames
    train_filename = f"{model_name}_train_{n_gram}_{corpus_name}.txt"
    test_filename = f"{model_name}_test_{n_gram}_{corpus_name}.txt"

    # Compute perplexities for train and test
    compute_perplexity(train_loader, train_filename)
    compute_perplexity(test_loader, test_filename)


def load_pretrained_model(lm_type, corpus, n_gram):
    """Load the pretrained model and vocabularies based on the given parameters."""
    print(lm_type,corpus,n_gram)
    model_dir = f"./Pretrained_Models/{lm_type}_{corpus}_n_{n_gram}"
    print(model_dir)
    if not os.path.exists(model_dir):
        print(f"Error: Pretrained model directory {model_dir} not found!")
        sys.exit(1)

    vocab_path = os.path.join(model_dir, "vocab.pkl")
    inv_vocab_path = os.path.join(model_dir, "inv_vocab.pkl")
    model_path = os.path.join(model_dir, "model.pt")

    if not all(os.path.exists(path) for path in [vocab_path, inv_vocab_path, model_path]):
        print("Error: Missing files in pretrained model directory!")
        sys.exit(1)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(inv_vocab_path, "rb") as f:
        inv_vocab = pickle.load(f)

    vocab_size = len(vocab)
    if lm_type == "ffnn":
        model = FFNNLanguageModel(vocab_size, embedding_dim=100, hidden_dim=200, n_gram=n_gram)
    elif lm_type == "rnn":
        model = RNNLanguageModel(vocab_size, embedding_dim=100, hidden_dim=200)
    elif lm_type == "lstm":
        model = LSTMLanguageModel(vocab_size, embedding_dim=100, hidden_dim=200)
    else:
        print("Invalid model type! Use '-f' for FFNN, '-r' for RNN, '-l' for LSTM")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model, vocab, inv_vocab

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 generator.py <lm_type> <corpus_path> <k>")
        sys.exit(1)

    lm_flag = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])

    model_map = {"-f": "ffnn", "-r": "rnn", "-l": "lstm"}
    if lm_flag not in model_map:
        print("Invalid model type! Use '-f' for FFNN, '-r' for RNN, '-l' for LSTM")
        sys.exit(1)

    lm_type = model_map[lm_flag]

    corpus_name = os.path.basename(corpus_path).replace(".txt", "").replace(" ", "_")

    n_gram=3
    model, vocab, inv_vocab = load_pretrained_model(lm_type, corpus_name, n_gram)
    print(f"Loaded pretrained model: {lm_type}_{corpus_name}_n_{n_gram}")

    while True:
        sentence = input("Input sentence: ").strip().split()
        predict_next_word(model, vocab, inv_vocab, n_gram, sentence, k)

if __name__ == "__main__":
    main()