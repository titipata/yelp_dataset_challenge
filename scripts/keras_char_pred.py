# simple character prediction script
# Keras model created by Zaw Htet

import pandas as pd
import numpy as np
from unidecode import unidecode
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding


yelp_review = pd.read_pickle('../data/yelp_academic_dataset_review.pickle')
yelp_bad = yelp_review.query("stars <= 2").query('votes_useful > 1')[0:100] # sample of bad review
yelp_bad_join = ' '.join(yelp_bad.text.map(unidecode))
chars = set(yelp_bad_join)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
yelp_bad_index = np.array([char_indices[s] for s in yelp_bad_join])


# get sequences to sequences input index characters
maxlen = 25
step = 3
sentences = []
next_chars = []
for i in range(0, len(yelp_bad_index) - maxlen):
    sentences.append(yelp_bad_index[i: i + maxlen])
    next_chars.append(yelp_bad_index[i + 1 : i + maxlen + 1])
print 'nb sequences:', len(sentences)


# input in numpy array format
X = np.array(sentences)
Y = np.zeros((len(sentences), maxlen, len(chars)))
for i, next_char in enumerate(next_chars):
    for t, char_id in enumerate(next_char):
        Y[i, t, char_id] = 1


# create keras model
n_chars = len(chars)
embedding_size = 128
print('Build model...')
model = Sequential()
model.add(Embedding(n_chars, embedding_size, input_length=maxlen))
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, embedding_size)))
model.add(Dropout(0.2))
# model.add(LSTM(512, return_sequences=True))
# model.add(Dropout(0.2))
model.add(TimeDistributedDense(n_chars))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# run model and save to keras_model folder
for iteration in range(1, 60):
    print '-' * 50
    print 'Iteration', iteration
    model.fit(X, Y, batch_size=128, nb_epoch=1)
    model_name = 'char_embed_test_%d.h5'%iteration
    if iteration % 10 == 0:
        model.save_weights(os.path.join('../keras_model/', model_name), overwrite=False)
        print 'model saved'
