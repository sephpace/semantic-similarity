"""
Author: Seph Pace
Email:  sephpace@gmail.com
"""

import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

from models import SimpleGloveEmbedding, MeanCenter


def visualize(seq1, seq2):
    """
    Visualizes a sentence in 2-dimensional space using GloVe embedding vectors.

    Points for each word are plotted in a scatter plot with the center point highlighted.

    Args:
        seq1 (str): The first sentence to visualize.
        seq2 (str): The second sentence to visualize.
    """
    # Load models
    print('Loading models...')
    glove = SimpleGloveEmbedding().eval()
    center = MeanCenter().eval()
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    # Get embeddings for the sentence
    seq1_data = glove(seq1)
    seq2_data = glove(seq2)
    data = torch.cat((seq1_data, seq2_data))

    # Calculate points
    print('Calculating points...')
    seq1_words = seq1.lower().split()
    seq2_words = seq2.lower().split()
    features = tsne.fit_transform(data)
    seq1_data = torch.from_numpy(features[:len(seq1_words), :])
    seq2_data = torch.from_numpy(features[len(seq1_words):, :])
    center1 = center(seq1_data.unsqueeze(0)).squeeze()
    center2 = center(seq2_data.unsqueeze(0)).squeeze()

    # Plot the graph
    print('Plotting the graph')
    plot_points(seq1_data, seq1_words, 'bo')
    plot_points(seq2_data, seq2_words, 'go')
    plt.plot(center1, center2, linestyle='--', marker='o', color='r', label='Center Distance')

    # Set up legend
    handles = (
        mpatches.Patch(color='blue', label='Sentence 1'),
        mpatches.Patch(color='green', label='Sentence 2'),
        mpatches.Patch(color='red', label='Center Points'),
    )
    plt.legend(handles=handles)

    # Save the graph
    plt.savefig('analysis/tsne.png')

    # Show the graph
    plt.show()


def plot_points(points, labels, point_type):
    """
    Plots the points on the scatter plot, connected by gray lines.

    points (np.array):    An array of size (seq_length, 2).
    labels (list of str): The labels for each point.
    point_type (str):     The point color and shape type.
    """
    for (x, y), label in zip(points, labels):
        plt.plot(x, y, point_type)
        plt.text(x, y + 5, label, fontsize=12)


if __name__ == '__main__':
    sentence1 = sys.argv[1]
    sentence2 = sys.argv[2]
    for s in (sentence1, sentence2):
        assert len(s) > 0, 'Sentence must have one or more characters!'
    visualize(sentence1, sentence2)