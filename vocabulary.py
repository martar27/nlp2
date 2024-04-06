from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    idx = 4  # Starting index for new words

    for data in dataset['train']:
        for word in word_tokenize(data['dialogue'].lower()):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
        for word in word_tokenize(data['summary'].lower()):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

if __name__ == "__main__":
    dataset = {'train': [{'dialogue': 'Hello, how are you?', 'summary': 'Greeting.'}]}
    vocab = build_vocab(dataset)
    print("Vocabulary size:", len(vocab))
