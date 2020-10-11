"""
Author: Seph Pace
Email:  sephpace@gmail.com
"""

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import TextSimilarityDataset
from models import SemanticSimilarity
from settings import EPOCHS, LEARNING_RATE, SHUFFLE, STATE_SAVE_PATH


def train():
    """
    Train the semantic similarity network.
    """
    # Load data
    print('Loading training data...')
    dataset = TextSimilarityDataset(train=True)
    data_loader = DataLoader(dataset, shuffle=SHUFFLE)

    # Load model and setup training parameters
    model = SemanticSimilarity().train()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = BCELoss()

    # Set up status template
    status_template = 'Epoch: {epoch:{epochs_str_len}d}/{epochs}  ' \
                      'Item: {step}/{data_len}  ' \
                      'Loss: {loss:.4f}'

    # Train the model
    print('Training model:')
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        for step, (seq1, seq2, target) in enumerate(data_loader):
            # Perform a training step
            optimizer.zero_grad()
            output = model(seq1[0], seq2[0])
            loss = criterion(output, target)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

            # Display status
            status = status_template.format(**{
                'epoch': epoch,
                'epochs': EPOCHS,
                'epochs_str_len': len(str(EPOCHS)),
                'step': step + 1,
                'data_len': len(data_loader),
                'loss': epoch_loss / (step + 1),
            })
            print(status, end='\r')
        print()

    # Save model state
    torch.save(model.state_dict(), STATE_SAVE_PATH)
    print(f'Model saved to: {STATE_SAVE_PATH}')


if __name__ == '__main__':
    train()
