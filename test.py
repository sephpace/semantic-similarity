"""
Author: Seph Pace
Email:  sephpace@gmail.com
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TextSimilarityDataset
from models import SemanticSimilarity, SimpleSemanticSimilarity
from settings import OUTPUT_SAVE_PATH


def test():
    """
    Test the semantic similarity model and against unseen data and compare it
    against the simple semantic similarity model's results on the same data.
    """
    # Load testing data
    print('Loading testing data...')
    dataset = TextSimilarityDataset()
    data_loader = DataLoader(dataset)

    # Set up the models
    model = SemanticSimilarity()
    model.load_state_dict(torch.load('states/semantic_similarity.pt'))
    model.eval()

    simple_model = SimpleSemanticSimilarity()
    simple_model.eval()

    # Set up output data dictionary
    output_data = {
        'sequence1': [],
        'sequence2': [],
        'simple_score': [],
        'attention_score': [],
    }

    # Set up the status template
    status_template = 'Item: {step}/{data_len}'

    # Test the model
    print('Evaluating model outputs:')
    for step, (seq1, seq2, _) in enumerate(data_loader):
        # Get model outputs
        output = model(seq1[0], seq2[0])
        simple_output = simple_model(seq1[0], seq2[0])

        # Round output to four decimal places
        decimal_places = 4
        multiplier = 10 ** decimal_places
        output = int((output.item() * multiplier) + 0.5) / multiplier
        simple_output = int((simple_output.item() * multiplier) + 0.5) / multiplier

        # Save output data
        output_data['sequence1'].append(seq1[0])
        output_data['sequence2'].append(seq2[0])
        output_data['simple_score'].append(simple_output)
        output_data['attention_score'].append(output)

        # Display status
        status = status_template.format(**{
            'step': step + 1,
            'data_len': len(data_loader),
        })
        print(status, end='\r')
    print()

    # Save model outputs
    data_frame = pd.DataFrame(output_data)
    data_frame.to_csv(OUTPUT_SAVE_PATH)
    print(f'Output saved to: {OUTPUT_SAVE_PATH}')


if __name__ == '__main__':
    test()
