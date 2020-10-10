"""
Author: Seph Pace
Email:  sephpace@gmail.com
"""

from settings import GLOVE_PATH


class Tokenizer:
    """
    Tokenizes sentences into sequences of tokens.

    Attributes:
        filter (str): Punctuation and other tokens to filter out.
        id_to_word (list): Converts ids to words.
        word_to_id (dict): Converts words to ids.
    """

    def __init__(self, path=GLOVE_PATH):
        """
        Constructor.
        """
        self.filter = '!()-[]{};:\'"\\,<>./?@#$%^&*_~'
        self.id_to_word = []
        self.word_to_id = {}

        # Get words from glove file
        with open(path, 'r') as file:
            for i, line in enumerate(file.readlines()):
                word = line.split()[0]
                self.id_to_word.append(word)
                self.word_to_id[word] = i

    def __call__(self, inputs):
        """
        Calls the convert function.

        Args:
            inputs (str or list of int): The inputs to convert.

        Returns:
            (list of int or str): The converted results.
        """
        return self.convert(inputs)

    def clean(self, sentence):
        """
        Cleans the sentence by removing punctuation and converting to lowercase.

        Args:
            sentence (str): The sentence to clean.

        Returns:
            (str): The cleaned sentence.
        """
        sentence = ''.join(c for c in sentence if c not in self.filter)
        return sentence.lower()

    def convert(self, inputs):
        """
        Converts strings to ids and ids to strings.

        Args:
            inputs (str or list of int): The inputs to convert.
        """
        if type(inputs) == str:
            return self.tokenize(inputs)
        else:
            return self.detokenize(inputs)

    def detokenize(self, ids):
        """
        Converts ids into a sentence.

        Args:
            ids (list of int): The list of ids to convert.

        Returns:
            (str): The detokenized string.
        """
        return ' '.join(self.id_to_word[i] for i in ids)

    def tokenize(self, sentence):
        """
        Converts a sentence into a list of ids.

        Args:
            sentence (str): The sentence to convert.

        Returns:
            (list of int): The converted ids.
        """
        words = self.clean(sentence).split()
        return [self.word_to_id.get(w, '') for w in words]