import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class CharTokenizer(Tokenizer):
    def __init__(self, alphabet, num_words=None, char_level=True,
                 oov_token='UNK'):
        super(CharTokenizer, self).__init__(num_words=num_words,
                                            char_level=char_level, oov_token=oov_token)
        # construct a new vocabulary
        self.alphabet = alphabet
        self.char_dict = {}
        for i, char in enumerate(self.alphabet):
            self.char_dict[char] = i + 1

        # Use char_dict to replace the tk.word_index
        self.word_index = self.char_dict.copy()
        # Add 'UNK' to the vocabulary
        self.word_index[self.oov_token] = max(self.char_dict.values()) + 1

    def tokenize(self, text):
        self.fit_on_texts(text)
        # Convert string to index
        sequences = self.texts_to_sequences(text)
        # Padding
        text = pad_sequences(sequences, maxlen=1014, padding='post')
        # Convert to numpy array
        text = np.array(text, dtype='float32')
        return text

    def create_embedding_weights(self):
        embedding_weights = []
        embedding_weights.append(np.zeros(len(self.word_index)))

        for char, i in self.word_index.items():         # from index 1 to 69
            onehot = np.zeros(len(self.word_index))
            onehot[i-1] = 1
            embedding_weights.append(onehot)
        return np.array(embedding_weights)
