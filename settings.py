"""
Global settings for the entire project.
"""

# Training parameters
EPOCHS = 100
LEARNING_RATE = 1e-5
SHUFFLE = True
STATE_SAVE_PATH = 'states/semantic_similarity.pt'

# Testing parameters
THRESHOLD = 0.95
OUTPUT_SAVE_PATH = 'analysis/output.csv'

# Glove file path
GLOVE_PATH = 'glove/glove.6B.50d.txt'
