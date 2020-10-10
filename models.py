"""
Author: Seph Pace
Email:  sephpace@gmail.com
"""

import torch
import torch.nn as nn

from settings import GLOVE_PATH


class SimpleGloveEmbedding(nn.Module):
    """
    A simple word to vector embedding mapping based on GloVe.

    Converts a sequence of words into an embedding tensor.

    Attributes:
        embedding_dict (dict): Word to embedding dictionary.
        embedding_dim (int):   The size of each embedding vector.
        unk_word (Tensor):     The default embedding for unknown words.
    """

    def __init__(self, path=GLOVE_PATH):
        """
        Constructor.

        Args:
            path (str): The path to a GloVe file.
        """
        super().__init__()

        # Fill embedding dictionary from glove file
        self.embedding_dict = {}
        with open(path, 'r') as file:
            for line in file.readlines():
                # Get word embedding
                tokens = line.split()
                word = tokens[0]
                vector = torch.Tensor([[float(n) for n in tokens[1:]]])
                self.embedding_dict[word] = vector

                # Set embedding dim size
                self.embedding_dim = vector.shape[-1]

        self.unk_word = torch.zeros(1, self.embedding_dim)

    def forward(self, seq):
        """
        Forward pass.

        Args:
            seq (str): The sequence of words to get the embedding of.
        """
        words = seq.lower().split()
        embedding = self.embedding_dict.get(words[0], self.unk_word)
        for word in words[1:]:
            new = self.embedding_dict.get(word, self.unk_word)
            embedding = torch.cat((embedding, new))
        return embedding


class MeanCenter(nn.Module):
    """
    Calculates the center (mean) of the given vector space using Euclidean geometry.

    Inputs are tensors containing a sequence of vectors in the shape (N, S, E).

    N = Batch size
    S = Sequence length
    E = Embedding dimensions
    """

    def forward(self, inputs):
        """
        Forward pass.

        Args:
            inputs (Tensor): Tensor of shape (N, S, E).

        Returns:
            (Tensor): The mean center (N, E).
        """
        return inputs.sum(1) / inputs.size(1)


class SemanticSimilarity(nn.Module):
    """
    Calculates the semantic similarity between two sequences of words
    by calculating the mean center that has been weighted with attention.
    """

    def __init__(self, path=GLOVE_PATH):
        """
        Constructor.

        Args:
            path (str): The path to a GloVe file.
        """
        super().__init__()
        self.embedding = SimpleGloveEmbedding(path)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding.embedding_dim, nhead=5)
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.mc = MeanCenter()

    def forward(self, seq1, seq2):
        """
        Forward pass.

        Args:
            seq1 (str): The first sequence.
            seq2 (str): The second sequence.

        Returns:
            (Tensor): A single-item tensor containing the semantic similarity
                      value.
        """
        # Find the center of each sequence in latent space
        embed1 = self.embedding(seq1).unsqueeze(0)
        embed1 = self.attention(embed1.transpose(0, 1))
        center1 = self.mc(embed1.transpose(1, 0).squeeze())

        embed2 = self.embedding(seq2).unsqueeze(0)
        embed2 = self.attention(embed2.transpose(0, 1))
        center2 = self.mc(embed2.transpose(1, 0).squeeze())

        # Find distance between both centers
        distance = center1.dist(center2)

        # Normalize and return output
        output = 1 - torch.relu(torch.tanh(distance))
        return output
