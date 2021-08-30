# import necessary libraries

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import conll2000

import seaborn as sns

from gensim.models import KeyedVectors

import tensorflow as tf  # tensorflow version == 2.6.0
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def read_data(path):
    table = pd.read_table(path).fillna('X')
    data = table.values 
    X, X_ = [], [] # store input sequence
    Y, Y_ = [], [] # store output sequence
    for word in data:
        if word[0] is not '.':         
            X_.append(word[0])  # entity[0] contains the word
            Y_.append(word[2])  # entity[1] contains corresponding tag
        else:        
            X_.append('.')
            Y_.append('PUNC')
            X.append(X_)
            Y.append(Y_)
            X_ = []
            Y_ = []
        
    return X,Y

path = 'C:/Users/dell/Desktop/nlp/NLAPOST2021-main/train/nr.gold.train'
# path_dev = 'C:/Users/dell/Desktop/nlp/NLAPOST2021-main/dev/nr.gold.dev'
X, Y = read_data(path)
# X_dev, Y_dev = read_data(path_dev)
# X.extend(X_dev)
# Y.extend(Y_dev)

################# Vectorize X and Y ################
# encode X
word_tokenizer = Tokenizer()                      # instantiate tokeniser
word_tokenizer.fit_on_texts(X)                    # fit tokeniser on data
X_encoded = word_tokenizer.texts_to_sequences(X)  # use the tokeniser to encode input sequence
# encode Y
tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(Y)
Y_encoded = tag_tokenizer.texts_to_sequences(Y)

# look at first encoded data point
# print("** Raw data point **", "\n", "-"*100, "\n")
# print('X: ', X[0], '\n')
# print('Y: ', Y[0], '\n')
# print()
# print("** Encoded data point **", "\n", "-"*100, "\n")
# print('X: ', X_encoded[0], '\n')
# print('Y: ', Y_encoded[0], '\n')

# make sure that each sequence of input and output is same length
different_length = [1 if len(input) != len(output) else 0 for input, output in zip(X_encoded, Y_encoded)]
print("{} sentences have disparate input-output lengths.".format(sum(different_length)))


################# Padding sequences ################
print("-"*50)
print('Begin Padding sequences')
# check length of longest sentence
lengths = [len(seq) for seq in X_encoded]
print("Length of longest sentence: {}".format(max(lengths)))

# Pad each sequence to MAX_SEQ_LENGTH using KERAS' pad_sequences() function. 
# Sentences longer than MAX_SEQ_LENGTH are truncated.
# Sentences shorter than MAX_SEQ_LENGTH are padded with zeroes.

# Truncation and padding can either be 'pre' or 'post'. 
# For padding we are using 'pre' padding type, that is, add zeroes on the left side.
# For truncation, we are using 'post', that is, truncate a sentence from right side.

MAX_SEQ_LENGTH = 100  # sequences greater than 100 in length will be truncated
X_padded = pad_sequences(X_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
Y_padded = pad_sequences(Y_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")

# print the first sequence
# print(X_padded[0], "\n"*3)
# print(Y_padded[0])

# assign padded sequences to X and Y
X, Y = X_padded, Y_padded


################# Word Embbeddings ################
print("-"*50)
print('Begin Word Embbeddings')
# word2vec
path = 'GoogleNews-vectors-negative300.bin.gz'
# load word2vec using the following function present in the gensim library
word2vec = KeyedVectors.load_word2vec_format(path, binary=True)

# assign word vectors from word2vec model
EMBEDDING_SIZE  = 300  # each word in word2vec model is represented using a 300 dimensional vector
VOCABULARY_SIZE = len(word_tokenizer.word_index) + 1

# create an empty embedding matix
embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))

# create a word to index dictionary mapping
word2id = word_tokenizer.word_index

# copy vectors from word2vec model to the words present in corpus
for word, index in word2id.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        pass

# check embedding dimension
print("Embeddings shape: {}".format(embedding_weights.shape))

# look at an embedding of a word
# print(embedding_weights[word_tokenizer.word_index['joy']])
# use Keras' to_categorical function to one-hot encode Y
Y = to_categorical(Y)
print(Y.shape)


################# Split data in training, validation and tesing sets ################
# split entire data into training and testing sets
TEST_SIZE = 0.15
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=4)
VALID_SIZE = 0.15
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=VALID_SIZE, random_state=4)

# print number of samples in each set
print("TRAINING DATA")
print('Shape of input sequences: {}'.format(X_train.shape))
print('Shape of output sequences: {}'.format(Y_train.shape))
print("-"*50)
print("VALIDATION DATA")
print('Shape of input sequences: {}'.format(X_validation.shape))
print('Shape of output sequences: {}'.format(Y_validation.shape))
print("-"*50)
print("TESTING DATA")
print('Shape of input sequences: {}'.format(X_test.shape))
print('Shape of output sequences: {}'.format(Y_test.shape))


################# create architecture ################
# total number of tags
NUM_CLASSES = Y.shape[2]
rnn_model = Sequential()

# create embedding layer - usually the first layer in text problems
rnn_model.add(Embedding(input_dim     =  VOCABULARY_SIZE,         # vocabulary size - number of unique words in data
                        output_dim    =  EMBEDDING_SIZE,          # length of vector with which each word is represented
                        input_length  =  MAX_SEQ_LENGTH,          # length of input sequence
                        weights       = [embedding_weights],      # word embedding matrix
                        trainable     =  True                     # True - update the embeddings while training
))

# add an RNN layer which contains 64 RNN cells
rnn_model.add(SimpleRNN(64, 
              return_sequences=True  # True - return whole sequence; False - return single output of the end of the sequence
))

# add time distributed (output at each sequence) layer
rnn_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))

# compile model
rnn_model.compile(loss      =  'categorical_crossentropy',
                  optimizer =  'adam',
                  metrics   =  ['acc'])

# check summary of the model
print(rnn_model.summary())
# fit model
rnn_training = rnn_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
# visualise training history
plt.plot(rnn_training.history['acc'])
plt.plot(rnn_training.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc="lower right")
plt.show()
plt.savefig('accuracy')