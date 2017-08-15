#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import configuration as config

from model import MemeModel

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='Training script\'s argument parser')
parser.add_argument('train_dir', help='directory where training checkpoints will be stored')
parser.add_argument('--train_inception', default=False, help='whether to train inception submodel variables')
parser.add_argument('--number_of_steps', default=1000000, help='number of training steps', type=int)
parser.add_argument('--log_every_n_steps', default=1, help='frequency at which loss and global step are logged', type=int)
parser.add_argument('--dataset_dir', default='meme_characters/', help='directory where memes are stored')
parser.add_argument('--vocab_file', default='meme_characters/word_count.txt', help='vocabulary file')
args = parser.parse_args()

if __name__ == '__main__':
    # Create training directory.
    train_dir = args.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = MemeModel('train', args.vocab_file, dataset_dir=args.dataset_dir)
        model.build()
        # Set up the learning rate.
        learning_rate_decay_fn = None
        # if train_inception:
        #     learning_rate = tf.constant(train_inception_learning_rate)
        # else:
        learning_rate = tf.constant(config.initial_learning_rate, name='learning_rate')
        if config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (config.num_examples_per_epoch /
                                     config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=config.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model.total_loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=config.optimizer,
            clip_gradients=config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=config.max_checkpoints_to_keep)
        # Run training.
        tf.logging.info('Training model') # meta_batch {}-{}...'.format(upper, lower))
        tf.contrib.slim.learning.train(
            train_op,
            args.train_dir,
            log_every_n_steps=args.log_every_n_steps,
            graph=g,
            global_step=model.global_step,
            number_of_steps=args.number_of_steps,
            saver=saver)
