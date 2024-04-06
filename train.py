import torch
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Attention  # Assuming PGNet is the main model class
from data_loader import load_and_preprocess_data
import torch.optim as optim
import torch.nn as nn
import os

def train_model(dataset, model, optimizer, criterion, num_epochs=10):
    best_val_loss = float('inf')
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

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for dialogues, summaries in DataLoader(dataset['validation'], batch_size=32, shuffle=False):
                output = model(dialogues, summaries)
                val_loss = criterion(output, summaries)
                total_val_loss += val_loss.item()
        print(f'Validation Loss: {total_val_loss}')

        # Check for best model
        if total_val_loss < best_val_loss:
            print('Saving best model')
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    datasets = load_and_preprocess_data()
    model = Encoder()  # Initialize your model here
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_model(datasets, model, optimizer, criterion)

# Test evaluation (optional here or can be in a separate script)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
total_test_loss = 0
with torch.no_grad():
    for dialogues, summaries in DataLoader(datasets['test'], batch_size=32, shuffle=False):
        output = model(dialogues, summaries)
        test_loss = criterion(output, summaries)
        total_test_loss += test_loss.item()
print(f'Test Loss: {total_test_loss}')
    model = PGNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_model(datasets, model, optimizer, criterion)

