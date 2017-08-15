#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import codecs
import numpy as np
import tensorflow as tf
import re

from utils.meme_stats import stats, sizeof_fmt
from utils.imagenet_utils import decode_predictions
from utils.inception_v3 import preprocess_input
from keras.preprocessing import image

# Some useful regexes
symbol_re = re.compile('[^\w\s\']')
space_re = re.compile('[\s\t\n]{2,}')

def clean_caption(caption):
    # Remove non-alphanumeric chars
    cleaned = re.sub(symbol_re, '', caption.lower())

    # Remove unnecessary spaces
    cleaned = re.sub(space_re, ' ', cleaned)

    return cleaned

def get_data(global_dir, model, vocab_file, quantity=1):
    '''
    Extracts the given quantity of the dataset contained in global_dir.
    If the needed amount of data is already in cache, it only extracts it from
    the path "cache/"
    Args:
      :global_dir, string: dataset global path (normally, "meme_characers/")
      :model: a Keras model that will be used to produce embeddings.
      :vocab_file, string: path to vocabulary file
      :quantity, int or float: proportion of data to be extracted. For instance,
      if quantity==1, then all the data will be considered.
    Returns:
      :caption_image, nparray: a mapping of caption and image embeddings
      :vocab_size, int: length of the list of all vocabulary words
    '''
    assert 0.0 < quantity and quantity <= 1.0
    assert os.path.exists(vocab_file)
    seqs_raw = []
    text = set()
    gdir_contents = [m for m in os.listdir(global_dir)
                     if os.path.isdir(os.path.join(global_dir, m))
                     and os.listdir(os.path.join(global_dir, m))]
    _, _, _, _, total_size = stats(global_dir, False)
    size = 0.0
    upper_bound = total_size * quantity
    count = 1
    for meme in gdir_contents:
        meme_file = os.path.join(global_dir, meme, '{}.csv'.format(meme))
        if os.path.exists(meme_file):
            img_path = os.path.join(global_dir, meme, '{}.jpg'.format(meme))
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            with codecs.open(meme_file, 'r') as f:
                reader = csv.reader(f)
                first = True
                for row in reader:
                    if not first:
                        cap = clean_caption(row[0])
                        if cap:
                            c = ['<S>'] + cap.split() + ['</S>']
                            text.update(c)
                            seqs_raw.append((preds, c))
                            size += len(cap.encode('utf-8'))
                    first = False
                    if size >= upper_bound:
                        break
                if size >= upper_bound:
                    break
            if size >= upper_bound:
                break

    with codecs.open(vocab_file, 'r') as f:
        word_indices = dict((c.split()[0], i) for i, c in enumerate(f))

    # Vectorization
    print('Vectorization...')
    cap_img = []
    for img, sentence in seqs_raw:
        cap_img.append([img, [word_indices[x] for x in sentence]])
    return np.array(cap_img), len(word_indices)

def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
    """Batches input images and captions.
    This function splits the caption into an input sequence and a target sequence,
    where the target sequence is the input sequence right-shifted by 1. Input and
    target sequences are batched and padded up to the maximum length of sequences
    in the batch. A mask is created to distinguish real words from padding words.
    Example:
    Actual captions in the batch ('-' denotes padded character):
    [
    [ 1 2 5 4 5 ],
    [ 1 2 3 4 - ],
    [ 1 2 3 - - ],
    ]
    input_seqs:
    [
    [ 1 2 3 4 ],
    [ 1 2 3 - ],
    [ 1 2 - - ],
    ]
    target_seqs:
    [
    [ 2 3 4 5 ],
    [ 2 3 4 - ],
    [ 2 3 - - ],
    ]
    mask:
    [
    [ 1 1 1 1 ],
    [ 1 1 1 0 ],
    [ 1 1 0 0 ],
    ]
    Args:
    images_and_captions: A list of pairs [image, caption], where image is a
    Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
    any length. Each pair will be processed and added to the queue in a
    separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.
    Returns:
    images: A Tensor of shape [batch_size, height, 1].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: An int32 Tensor of shape [batch_size, padded_length].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
    """
    print('images_and_captions:', np.shape(images_and_captions))
    print('batch_size:', batch_size)
    print('queue_capacity:', queue_capacity)
    enqueue_list = []
    for image, caption in images_and_captions:
        caption_length = tf.shape(caption)[0]
        input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
        input_seq = tf.slice(caption, [0], input_length)
        target_seq = tf.slice(caption, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)
        img = image.flatten()
        enqueue_list.append([img, input_seq, target_seq, indicator])
        # print(count, np.shape(image), np.shape(caption))

    images, input_seqs, target_seqs, mask = tf.train.batch_join(
        enqueue_list,
        batch_size=batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        name="batch_and_pad")
    if add_summaries:
        lengths = tf.add(tf.reduce_sum(mask, 1), 1)
        tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
        tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
        tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))
    return images, input_seqs, target_seqs, mask
