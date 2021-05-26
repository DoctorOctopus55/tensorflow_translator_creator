import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, GRU, Embedding
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class TokenizerWrap(Tokenizer):
    def __init__(self, texts, padding, reverse=False, num_words=None):
        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)

        self.index_to_word = dict(zip(self.word_index.values(), self.word_index.keys()))
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            self.tokens = [list(reversed(x)) for x in self.tokens]
            truncating = 'pre'
        
        else:
            truncating = 'post'
        
        self.num_tokens = [len(x) for x in self.tokens]
        self.max_tokens = np.mean(self.num_tokens) + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)

        self.tokens_padded = pad_sequences(self.tokens, maxlen=self.max_tokens, padding=padding, truncating=truncating)
    
    def token_to_word(self, token):
        word = ' ' if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token] for token in tokens if token != 0]
        text = ' '.join(words)
        return text

    def text_to_tokens(self, text, padding, reverse=False):
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            tokens = np.flip(tokens, axis=1)
            truncating = 'pre'

        else:
            truncating = 'post'

        tokens = pad_sequences(tokens, maxlen=self.max_tokens, padding=padding, truncating=truncating)

        return tokens
