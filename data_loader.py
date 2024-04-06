from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import torch
import nltk

nltk.download('punkt')

class DialogueSummaryDataset(Dataset):
    def __init__(self, dialogues, summaries, vocab):
        self.dialogues = dialogues
        self.summaries = summaries
        self.vocab = vocab

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = [self.vocab.get(word, self.vocab['<unk>']) for word in word_tokenize(self.dialogues[idx].lower())]
        summary = [self.vocab.get(word, self.vocab['<unk>']) for word in word_tokenize(self.summaries[idx].lower())]
        return torch.tensor(dialogue, dtype=torch.long), torch.tensor(summary, dtype=torch.long)

def build_vocab(dataset):
    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    idx = 4
    for split in ['train', 'test', 'validation']:
        for item in dataset[split]:
            words = word_tokenize(item['dialogue'].lower()) + word_tokenize(item['summary'].lower())
            for word in words:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
    return vocab

def load_and_preprocess_data():
    dataset = load_dataset('samsum')
    vocab = build_vocab(dataset)
    processed_datasets = {}
    for split in ['train', 'test', 'validation']:
        dialogues = [item['dialogue'] for item in dataset[split]]
        summaries = [item['summary'] for item in dataset[split]]
        processed_datasets[split] = DialogueSummaryDataset(dialogues, summaries, vocab)
    return processed_datasets

if __name__ == "__main__":
    datasets = load_and_preprocess_data()
    for split, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for batch in loader:
            print(batch)
    dataset = load_dataset("samsum")
    vocab = build_vocab(dataset)
    processed_datasets = {}
    for split in ['train', 'test', 'validation']:
        dialogues = [item['dialogue'] for item in dataset[split]]
        summaries = [item['summary'] for item in dataset[split]]
        processed_datasets[split] = DialogueSummaryDataset(dialogues, summaries, vocab)
    return processed_datasets

if __name__ == "__main__":
    datasets = load_and_preprocess_data()
    for split, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f'Processing {split} data:')
        for batch in loader:
            print(batch)

if __name__ == "__main__":
    dataset = load_and_preprocess_data()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in loader:
        print(batch)

