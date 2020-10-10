"""
Author: Seph Pace
Email:  sephpace@gmail.com
"""

import torch
import torch.nn as nn


class SimpleGloveEmbedding(nn.Module):
    """
    A simple word to vector embedding mapping based on GloVe.

    Attributes:
        embedding_dict (dict): Word to embedding dictionary.
        embedding_dim (int):   The size of each embedding vector.
        unk_word (Tensor):     The default embedding for unknown words.
    """

    def __init__(self, path='glove/glove.6B.50d.txt'):
        """
        Constructor.

        Args:
            path (str): The path to a GloVe file.
        """
        super().__init__()
        self.embedding_dim = 300
        self.unk_word = torch.zeros(self.embedding_dim)


        # Fill embedding dictionary from glove file
        self.embedding_dict = {}
        with open(path, 'r') as file:
            for line in file.readlines():
                tokens = line.split()
                word = tokens[0]
                vector = torch.Tensor([float(n) for n in tokens[1:]])
                self.embedding_dict[word] = vector

    def forward(self, word):
        """
        Forward pass.

        Args:
            word (str): The word to get the embedding for.
        """
        return self.embedding_dict.get(word, self.unk_word)
