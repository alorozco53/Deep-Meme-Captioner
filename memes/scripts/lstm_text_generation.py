'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from meme_stats import sizeof_fmt
import numpy as np
import random
import sys
import os
import csv
import codecs
import argparse
import re


# # default training corpus
# path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
# text = open(path).read().lower()
# print('corpus length:', len(text))

parser = argparse.ArgumentParser(description='script\'s argument parser')
parser.add_argument('meme_characs_path', help='directory where memes are stored')
parser.add_argument('--max', help='maximum size (in bytes!) of memes to be used',
                    default=588000)
args = parser.parse_args()
global_dir = args.meme_characs_path
print('extracting', sizeof_fmt(float(args.max)), 'of memes from', global_dir)
text = ''
gdir_contents = [m for m in os.listdir(global_dir)
                 if os.path.isdir(os.path.join(global_dir, m))
                 and os.listdir(os.path.join(global_dir, m)) != []]
counter = 1
two_spaces = re.compile('[ ]{2,}')
while len(text.encode('utf-8')) < float(args.max):
    for meme in gdir_contents:
        meme_file = os.path.join(global_dir, meme, '{}.csv'.format(meme))
        if os.path.exists(meme_file):
            with codecs.open(meme_file, 'r') as f:
                reader = csv.reader(f)
                i = 0
                for row in reader:
                    if i == counter:
                        clean = '{}. '.format(row[0].strip()).lower()
                        clean = two_spaces.sub(' ', clean)
                        text += clean
                        break
                    i += 1
    counter += 1

# halved_contents = gdir_contents[:len(halved_contents)/4]
# for meme in halved_contents:
#     meme_file = os.path.join(global_dir, meme, '{}.csv'.format(meme))
#     if os.path.exists(meme_file):
#         with codecs.open(meme_file, 'r') as f:
#             reader = csv.reader(f)
#             first = True
#             for row in reader:
#                 if not first:
#                     text += '{} '.format(row[0])
#                 first = False
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
