import numpy as np
from tensorflow.keras.layers import TextVectorization


class CharTokenizer(TextVectorization):
    def __init__(
            self,
            alphabet,                           # equal to legacy Tokenizer.filters
            split="character",
            output_sequence_length=1014,
            standardize=None                    # we set to None because otherwise it would remove special characters
    ):
        super(
            CharTokenizer, self).__init__(
            split=split,
            output_sequence_length=output_sequence_length,
            standardize=standardize
        )

    def tokenize(self, text):
        # apart from one extra character '' (index 0) in the vocabulary,
        # the resulting text vectors are practically the same as in legacy Tokenizer class
        self.adapt(text)
        self.word_index = self.get_vocabulary()     # equal to legacy Tokenizer.word_index
        return self(text)

    def create_embedding_weights(self):
        embedding_weights = []
        embedding_weights.append(np.zeros(self.vocabulary_size()))

        for i in range(self.vocabulary_size()):         # from index 1 to 70
            onehot = np.zeros(self.vocabulary_size())
            onehot[i-1] = 1
            embedding_weights.append(onehot)
        return np.array(embedding_weights)
