import torch
from torch.utils.data import DataLoader
from model import PGNet, Encoder, Decoder, Attention
from data_loader import load_and_preprocess_data
import torch.optim as optim
import torch.nn as nn

def train_model(dataset, model, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for dialogues, summaries in DataLoader(dataset['train'], batch_size=32, shuffle=True):
            optimizer.zero_grad()
            output = model(dialogues, summaries)
            loss = criterion(output, summaries)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss}')

if __name__ == '__main__':
    datasets = load_and_preprocess_data()
    model = PGNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_model(datasets, model, optimizer, criterion)

