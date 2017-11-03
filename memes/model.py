#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import numpy as np
import tensorflow as tf
import configuration as config

# The `stats` function will compute the statistics from the existing dataset in
# [`meme_characters/`](meme_characters/).
from utils.meme_stats import stats, sizeof_fmt

from utils.data_utils import get_data, batch_with_dynamic_pad
from utils.image_processing import process_image
from utils.imagenet_utils import get_feature_layer
from keras.models import load_model, Model
from keras.applications.inception_v3 import InceptionV3

# A [`LSTMStateTuple`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple)
# will be needed to store all the (_batched_) embeddings obtained from the ConvNet, in order to initialize our RecurrentNet.
# from tensorflow.contrib.rnn import LSTMStateTuple

class MemeModel(object):
    """
    Image-to-text implementation based on (http://arxiv.org/abs/1411.4555).

    Keras Inception - Tensorflow LSTM

    See [`inception_v3.py`](utils/inception_v3.py) for more info on the specification.
    The two last layers are ignored due to implementation issues
    """

    def __init__(self,
                 mode,
                 vocab_file,
                 dataset_dir='meme_characters/',
                 model_file='inception_log3.0/fine_inception.h5',
                 cap_per_img='ALL'):
        assert mode in ['train', 'eval', 'inference']
        # assert os.path.exists(vocab_file)
        self.mode = mode
        self.vocab_file = vocab_file
        self.dataset_dir = dataset_dir
        self.model_file = model_file
        self.cpi = cap_per_img

    def build_conv_model(self, image_format='jpeg'):
        """
        Builds the convolutional neural network (Keras)
        """
        # self.model = InceptionV3(include_top=False,
        #                          weights=config.inception_weights,
        #                          input_shape=(150, 150, 3))
        base_model = load_model(self.model_file)
        base_model.layers.pop()
        base_model.outputs = [base_model.layers[-1].output]
        base_model.layers[-2].outbound_nodes = []
        self.model = Model(base_model.input, base_model.layers[-1].output)
        if self.mode == 'train':
            print(self.model.summary())
            print(self.model.output_shape)
        elif self.mode == 'inference':
            shape = self.model.output.shape.as_list()[-1]
            images = tf.placeholder(tf.float32, shape=(1, shape), name='image_feed')
            # In inference mode, images and inputs are fed via placeholders.
            # image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            input_feed = tf.placeholder(dtype=tf.int64,
                                        shape=[None],  # batch_size
                                        name="input_feed")

            # # Process image and insert batch dimensions.
            # images = tf.expand_dims(self.process_image(image_feed, image_format), 0)
            input_seqs = tf.expand_dims(input_feed, 1)

            # No target sequences or input mask in inference mode.
            target_seqs = None
            input_mask = None

            self.images = images
            self.input_seqs = input_seqs
            self.target_seqs = target_seqs
            self.input_mask = input_mask


    def is_training(self):
        return self.mode == 'train'

    def print_dataset_stats(self, verbose=False):
        """
        A data sample will extracted from [`meme_characters/`](meme_characters/). This info must
        be crawled!
        """
        ncaptions, nmeme_characters, nwords, nchars, total_size = stats(self.dataset_dir, verbose)
        print('total number of captions', ncaptions)
        print('total number of meme characters', nmeme_characters)
        print('total number of words:', nwords)
        print('total number of characters:', nchars)
        print('total size:', sizeof_fmt(total_size))

    def process_image(self, encoded_image, image_format='jpeg', thread_id=0):
        """Decodes and processes an image string.
        Args:
        encoded_image: A scalar string Tensor; the encoded image.
        thread_id: Preprocessing thread id used to select the ordering of color
        distortions.
        Returns:
        A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        assert image_format in ['jpeg', 'png']
        return process_image(encoded_image,
                             is_training=self.is_training(),
                             height=config.image_height,
                             width=config.image_width,
                             thread_id=thread_id,
                             image_format=image_format)

    def build_inputs(self):
        """
        Now we get a proportion of the data and we encode it using a CNN and sequence embeddings.
        """
        if self.mode != 'inference':
            print('Getting dataset from {}'.format(self.dataset_dir))
            self.images_captions, self.vocab_size = get_data(self.dataset_dir, self.model, self.vocab_file,
                                                             quantity=config.dataset_proportion,
                                                             captions_per_image=self.cpi)
            self.vocab_size += 1
            print(np.shape(self.images_captions))
        else:
            with codecs.open(self.vocab_file, 'r') as f:
                self.vocab_size = len([l for l in f]) + 1


    def init_model_params(self):
        print('Initializing the model\'s parameters...')

        embedding_size = config.embedding_size
        # To match the "Show and Tell" paper we initialize all variables with a
        # random uniform initializer.
        initializer_scale = config.initializer_scale
        self.initializer = tf.random_uniform_initializer(
            minval=-initializer_scale,
            maxval=initializer_scale)

        if self.mode != 'inference':
            queue_capacity = (2 * config.num_preprocess_threads *
                              config.batch_size)
            mbs = config.meta_batch_size
            col_size = np.shape(self.images_captions)[0]
            self.images, self.input_seqs, self.target_seqs, self.input_mask = (
                batch_with_dynamic_pad(self.images_captions,# [:5000],
                                       batch_size=config.batch_size,
                                       queue_capacity=queue_capacity))

        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.embedding_map = tf.get_variable(
                name="map",
                shape=[self.vocab_size, embedding_size],
                initializer=self.initializer)
            self.seq_embeddings = tf.nn.embedding_lookup(self.embedding_map, self.input_seqs)

    def build_img_embeddings(self):
        """
        Map inception output into embedding space.
        """
        # if self.mode == 'inference':
        #     self.images = tf.placeholder(tf.float32, shape=(1, 1000), name='input')
        print('Mapping image embeddings...')
        print(self.images.get_shape())
        with tf.variable_scope("image_embedding") as scope:
            self.image_embeddings = tf.contrib.layers.fully_connected(
                inputs=self.images,
                num_outputs=config.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)

        # Save the embedding size in the graph.
        tf.constant(config.embedding_size, name="embedding_size")

    def build_lstm(self):
        """
        LSTM Specification
        A (_training_) LSTM net is specified given a [`BasicLSTMCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell), a `LSTMStateTuple`
        """
        print('Building the LSTM model...')
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=config.num_units,
                                                 state_is_tuple=True)
        if self.mode == "train":
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell,
                input_keep_prob=config.lstm_dropout_keep_prob,
                output_keep_prob=config.lstm_dropout_keep_prob)

        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
            zero_state = lstm_cell.zero_state(
                batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
            print(self.image_embeddings.get_shape())
            _, initial_state = lstm_cell(self.image_embeddings, zero_state)

            # Allow the LSTM variables to be reused.
            lstm_scope.reuse_variables()
            if self.mode == "inference":
                # In inference mode, use concatenated states for convenient feeding and
                # fetching.
                tf.concat(axis=1, values=initial_state, name="initial_state")

                # Placeholder for feeding a batch of concatenated states.
                state_feed = tf.placeholder(dtype=tf.float32,
                                            shape=[None, sum(lstm_cell.state_size)],
                                            name="state_feed")
                state_tuple = tf.split(state_feed, 2, axis=1)
                # Run a single LSTM step.
                self.lstm_outputs, state_tuple = lstm_cell(
                    inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
                    state=state_tuple)

                # Concatentate the resulting state.
                tf.concat(axis=1, values=state_tuple, name="state")
            else:
                # Run the batch of sequence embeddings through the LSTM.
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                self.lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                         inputs=self.seq_embeddings,
                                                         sequence_length=sequence_length,
                                                         initial_state=initial_state,
                                                         dtype=tf.float32,
                                                         scope=lstm_scope)

        # Stack batches vertically.
        self.lstm_outputs = tf.reshape(self.lstm_outputs, [-1, lstm_cell.output_size])

        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=self.lstm_outputs,
                num_outputs=self.vocab_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logits_scope)

        if self.mode == "inference":
            tf.nn.softmax(logits, name="softmax")
        else:
            targets = tf.reshape(self.target_seqs, [-1])
            weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

            # Compute losses.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                    logits=logits)
            batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                                tf.reduce_sum(weights),
                                name="batch_loss")
            tf.contrib.losses.add_loss(batch_loss)
            total_loss = tf.contrib.losses.get_total_loss()

            # Add summaries.
            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            if self.mode == 'eval':
                sess = tf.Session()
                with sess.as_default():
                    sess.run(tf.global_variables_initializer())
                    for var in tf.trainable_variables():
                        tf.summary.histogram("parameters/" + var.op.name, var.eval(session=sess))

            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.


    # def setup_inception_initializer(self):
        # Sets up the function to restore inception variables from checkpoint.
        # if mode != "inference":
        # Restore inception variables only.
        # inception_variables = tf.get_collection(
        #     tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
        # saver = tf.train.Saver(inception_variables)

        # def restore_fn(sess):
        #     tf.logging.info("Restoring Inception variables from checkpoint file %s",
        #                     inception_checkpoint_file)
        #     saver.restore(sess, inception_checkpoint_file)

        # init_fn = restore_fn
        # else:
        #     init_fn = None
        # pass


    def setup_global_step(self):
        # Sets up the global step Tensor.
        print('Setting up the global step tensor...')
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self, image_format='jpeg'):
        self.build_conv_model(image_format)
        # self.print_dataset_stats()
        self.build_inputs()
        # for upper_bound, lower_bound in self.init_model_params():
        self.init_model_params()
        self.build_img_embeddings()
        self.build_lstm()
        self.setup_global_step()
        # yield upper_bound, lower_bound

if __name__ == '__main__':
    MemeModel('eval').build()
