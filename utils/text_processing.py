import numpy as np
import random
import nltk
from nltk.corpus import wordnet as wn
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


def get_synonyms(word):
    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def replace_words_with_synonyms(text, p=0.5, q=0.5):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    
    # Extract replaceable words (nouns, verbs, adjectives, and adverbs)
    replaceable_words = [word for word, pos in nltk.pos_tag(words) if pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ') or pos.startswith('RB')]
    
    # Randomly choose a word replacement count 'r' using a geometric distribution with parameter 'p'
    r = np.random.geometric(p)
    
    for _ in range(r):
        if len(replaceable_words) > 0:
            # Randomly choose the index 's' of the synonym using a geometric distribution with parameter 'q'
            s = np.random.geometric(q)
            s = min(s, len(replaceable_words) - 1)
            
            word_to_replace = random.choice(replaceable_words)
            synonyms = get_synonyms(word_to_replace)
            
            if synonyms:
                # Replace the selected word with the synonym
                new_word = random.choice(synonyms)
                text = text.replace(word_to_replace, new_word, 1)
                
                # Remove the word from the list of replaceable words to avoid replacing it again
                replaceable_words.remove(word_to_replace)
    
    return text