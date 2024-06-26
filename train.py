import torch
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Attention  # Assuming PGNet is the main model class
from data_loader import load_and_preprocess_data
import torch.optim as optim
import torch.nn as nn
import os

num_epochs = 3

def train_model(dataset, model, optimizer, criterion, num_epochs=num_epochs):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for dialogues, summaries, input_lengths in DataLoader(dataset['train'], batch_size=8, shuffle=True):
            optimizer.zero_grad()
            output = model(dialogues, input_lengths)
            loss = criterion(output, summaries)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss}')

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for dialogues, summaries, input_lengths in DataLoader(dataset['validation'], batch_size=8, shuffle=False):
                output = model(dialogues, input_lengths)
                val_loss = criterion(output, summaries)
                total_val_loss += val_loss.item()
        print(f'Validation Loss: {total_val_loss}')

        # Check for best model
        if total_val_loss < best_val_loss:
            print('Saving best model')
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    from data_loader import load_and_preprocess_data, collate_fn

    datasets = load_and_preprocess_data()
    # Assuming vocab is stored as an attribute of the dataset. Adjust if it's stored differently.
    vocab_size = len(datasets['train'].vocab)  
    hidden_size = 256  # Example size, adjust based on your needs

    model = Encoder(input_size=vocab_size, hidden_size=hidden_size)  # Initialize your model here
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = datasets['train']
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for dialogues, summaries, input_lengths in train_loader:
            optimizer.zero_grad()
            output = model(dialogues, input_lengths)
            loss = criterion(output, summaries)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss}')
    train_model(datasets, model, optimizer, criterion)

# Test evaluation (optional here or can be in a separate script)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
total_test_loss = 0
with torch.no_grad():
    for dialogues, summaries, input_lengths in DataLoader(datasets['test'], batch_size=32, shuffle=False):
        output = model(dialogues, input_lengths)
        test_loss = criterion(output, summaries)
        total_test_loss += test_loss.item()
print(f'Test Loss: {total_test_loss}')
model = PGNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_model(datasets, model, optimizer, criterion)

