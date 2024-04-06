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

def load_and_preprocess_data():
    dataset = load_dataset("samsum")
    dialogues = [item['dialogue'] for item in dataset['train']]
    summaries = [item['summary'] for item in dataset['train']]
    # Vocabulary building should be done here
    vocab = {}  # Placeholder for actual vocabulary building process
    return DialogueSummaryDataset(dialogues, summaries, vocab)

if __name__ == "__main__":
    dataset = load_and_preprocess_data()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in loader:
        print(batch)

