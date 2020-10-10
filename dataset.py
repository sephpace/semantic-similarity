"""
The Text Similarity dataset by Rishi Sankineni.  CSV files can be
downloaded from https://www.kaggle.com/rishisankineni/text-similarity/

Implemented in Pytorch by Seph Pace.
"""

import pandas as pd
from torch.utils.data import Dataset


class TextSimilarityDataset(Dataset):
    """
    The text similiarity dataset is a list of product ids (product X an product
    Y) with descriptions X and Y respectively.  They also have identifiers
    called tickers (ticker_x, ticker_y).  If the tickers match, they are
    semantically the same.

    Data is stored in a list of tuples of the two sequences followed by the
    target label (True for same and False otherwise).

    Attributes:
        data (list of (str, str, bool)): The data in the dataset (seq1, seq2, target)
    """

    def __init__(self, train=False):
        """
        Constructor.  Loads the data from csv files.

        Will load the testing dataset by default and will load the training
        dataset if explicitly stated in the train parameter.

        Args:
            train (bool): True if the training dataset should be loaded.
        """
        # Load the data frame
        if train:
            data_frame = pd.read_csv('data/train.csv')
        else:
            data_frame = pd.read_csv('data/test.csv')

        # Only grab the descriptions (sequences) and the same security (target)
        columns = ('description_x', 'description_y', 'same_security')

        # Combine the data into tuples
        self.data = list(zip(*(data_frame[col] for col in columns)))


    def __getitem__(self, idx):
        """
        Get the sequences and target at the specified index.

        Args:
            idx (int): The index of the data tuple.

        Returns:
            (str, str, bool): The sequences and the target.
        """
        return self.data[idx]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            (int): The length of the dataset.
        """
        return len(self.data)
