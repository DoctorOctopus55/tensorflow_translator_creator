import tensorflow as tf
import numpy as np

from tokenizerwrap import TokenizerWrap
from keras.models import Model
from keras.layers import Input, Dense, GRU, Embedding
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tf.compat.v1.disable_eager_execution()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class Data():
    def __init__(self):
        self.mark_start = 'ssss '
        self.mark_end = ' eeee'

        self.data_src = []
        self.data_dest = []

        self.read_text_data('tur.txt')

        self.tokenizer_src = TokenizerWrap(texts=self.data_src, 
                                padding='pre',
                                reverse=True,
                                num_words=None)


        self.tokenizer_dest = TokenizerWrap(texts=self.data_dest,
                                padding='post',
                                reverse=False,
                                num_words=None) 
        
        self.token_start = self.tokenizer_dest.word_index[self.mark_start.strip()]
        self.token_end = self.tokenizer_dest.word_index[self.mark_end.strip()]

        self.tokens_src = self.tokenizer_src.tokens_padded
        self.tokens_dest = self.tokenizer_dest.tokens_padded

        self.encoder_input_data = self.tokens_src

        self.decoder_input_data = self.tokens_dest[:, :-1]
        self.decoder_output_data = self.tokens_dest[:, 1:]

        self.num_encoder_words = len(self.tokenizer_src.word_index)
        self.num_decoder_words = len(self.tokenizer_dest.word_index)

        self.embedding_size = 100

        self.read_glove_words('glove.6B.100d.txt')


    def read_text_data(self, dataset, encoding='UTF-8', separater='\t'):
        for line in open(dataset, encoding=encoding):
            self.en_text, self.tr_text = line.rstrip().split(separater)

            self.tr_text = self.mark_start + self.tr_text + self.mark_end

            self.data_src.append(self.en_text)
            self.data_dest.append(self.tr_text)

    def read_glove_words(self, dataset, encoding='UTF-8'):
        self.word2vec = {}
        with open(dataset, encoding=encoding) as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                self.word2vec[word] = vec
        
        self.embedding_matrix = np.random.uniform(-1, 1, (self.num_encoder_words, self.embedding_size))

        for word, i in self.tokenizer_src.word_index.items():
            if i < self.num_encoder_words:
                embedding_vector = self.word2vec.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector


class Encoder(Data):
    def __init__(self):
        super().__init__()
        self.encoder_input = Input(shape=(None,), name='encoder_input')

        self.encoder_embedding = Embedding(input_dim=self.num_encoder_words,
                              output_dim=self.embedding_size,
                              weights=[self.embedding_matrix],
                              trainable=True,
                              name='encoder_embedding')
        self.state_size = 256

        self.encoder_gru1 = tf.compat.v1.keras.layers.CuDNNGRU(self.state_size, name='encoder_gru1', return_sequences=True)
        self.encoder_gru2 = tf.compat.v1.keras.layers.CuDNNGRU(self.state_size, name='encoder_gru2', return_sequences=True)
        self.encoder_gru3 = tf.compat.v1.keras.layers.CuDNNGRU(self.state_size, name='encoder_gru3', return_sequences=False)
        
        self.encoder_output = self.connect_encoder()

    def connect_encoder(self):
        net = self.encoder_input
        
        net = self.encoder_embedding(net)
        
        net = self.encoder_gru1(net)
        net = self.encoder_gru2(net)
        net = self.encoder_gru3(net)
        
        encoder_output = net
        
        return encoder_output


class Decoder(Encoder):
    def __init__(self):
        super().__init__()

        self.decoder_initial_state = Input(shape=(self.state_size,), name='decoder_initial_state')
        
        self.decoder_input = Input(shape=(None,), name='decoder_input')

        self.decoder_embedding = Embedding(input_dim=self.num_decoder_words,
                                    output_dim=self.embedding_size,
                                    name='decoder_embedding')
        
        self.decoder_gru1 = tf.compat.v1.keras.layers.CuDNNGRU(self.state_size, name='decoder_gru1', return_sequences=True)
        self.decoder_gru2 = tf.compat.v1.keras.layers.CuDNNGRU(self.state_size, name='decoder_gru2', return_sequences=True)
        self.decoder_gru3 = tf.compat.v1.keras.layers.CuDNNGRU(self.state_size, name='decoder_gru3', return_sequences=True) 

        self.decoder_dense = Dense(self.num_decoder_words,
                      activation='linear',
                      name='decoder_output')
        
        self.decoder_output = self.connect_decoder(initial_state=self.encoder_output)
        
    def connect_decoder(self, initial_state):
        net = self.decoder_input
        
        net = self.decoder_embedding(net)
        
        net = self.decoder_gru1(net, initial_state=initial_state)
        net = self.decoder_gru2(net, initial_state=initial_state)
        net = self.decoder_gru3(net, initial_state=initial_state)
        
        decoder_output = self.decoder_dense(net)
        
        return decoder_output


class TModel(Decoder):
    def __init__(self):
        super().__init__()

        self.model_train = Model(inputs=[self.encoder_input, self.decoder_input], outputs=[self.decoder_output])
        self.model_encoder = Model(inputs=[self.encoder_input], outputs=[self.encoder_output])

        self.decoder_output = self.connect_decoder(initial_state=self.decoder_initial_state)

        self.model_decoder = Model(inputs=[self.decoder_input, self.decoder_initial_state], outputs=[self.decoder_output])
        self.decoder_target = tf.compat.v1.placeholder(dtype='int32', shape=(None, None))

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

        self.model_train.compile(optimizer=self.optimizer,
                    loss=self.sparse_cross_entropy,
                    target_tensors=[self.decoder_target])

        self.path_checkpoint = 'checkpointss.keras'
        self.checkpoint = ModelCheckpoint(filepath=self.path_checkpoint, save_weights_only=True)

        self.x_data = {'encoder_input': self.encoder_input_data, 'decoder_input': self.decoder_input_data}
        self.y_data = {'decoder_output': self.decoder_output_data}

        self.train(epoch=2, batchsize=256)

    def sparse_cross_entropy(self, y_true, y_pred):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        loss_mean = tf.reduce_mean(loss)

        return loss_mean
    
    def train(self, epoch, batchsize):
        try:
            self.model_train.load_weights(self.path_checkpoint)

        except Exception as error:
            print(error)
            print('Checkpoint couldnt lodaded. Train starts...')

            self.model_train.fit(x=self.x_data,
                y=self.y_data,
                batch_size=batchsize,
                epochs=epoch,
                callbacks=[self.checkpoint])
        
    def translate(self, input_text, true_output_text=None, returner=False):
        input_tokens = self.tokenizer_src.text_to_tokens(text=input_text,
                                                    reverse=True,
                                                    padding='pre')
        
        initial_state = self.model_encoder.predict(input_tokens)
        
        max_tokens = self.tokenizer_dest.max_tokens
        
        decoder_input_data = np.zeros(shape=(1, max_tokens), dtype=np.int)
        
        token_int = self.token_start
        output_text = ''
        count_tokens = 0
        
        while token_int != self.token_end and count_tokens < max_tokens:
            decoder_input_data[0, count_tokens] = token_int
            self.x_data = {'decoder_initial_state': initial_state, 'decoder_input': decoder_input_data}
            
            self.decoder_output = self.model_decoder.predict(self.x_data)
            
            token_onehot = self.decoder_output[0, count_tokens, :]
            token_int = np.argmax(token_onehot)
            
            sampled_word = self.tokenizer_dest.token_to_word(token_int)
            output_text += ' ' + sampled_word
            count_tokens += 1
        
        if returner:
            return output_text
        
        else:
            print('Input text: ' + str(input_text))
            print()
            
            print('Translated text: ' + str(output_text))
            print('------------------------------------------------------------------')
            
            if true_output_text is not None:
                print('True output text: ' + str(true_output_text))
                print()

    def talk(self):
        talker = True
        while True:
            input_1 = input('Please enter something in english: ')
            if input_1 == 'exit' or 'Exit' or 'EXIT':
                talker = False
            print('-------------------------------------------------------------------')
            self.translate(input_text=input_1)


def run():
    translator_model = TModel()
    translator_model.talk()
run()