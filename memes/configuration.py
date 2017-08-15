#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


"""Configuration parameters and default values
"""

# InceptionV3 weights
inception_weights = 'imagenet'

# Dimensions of Inception v3 input images.
image_height = 299
image_width = 299

# Proportion of data to be gathered from dataset
# in the range (0, 1]
dataset_proportion = 1 # 0.25

# Initializer varibles according to the "Show and Tell" paper
initializer_scale = 0.08
batch_size = 32 # 100
queue_capacity = 200
meta_batch_size = 200
vocab_size = 500 # 12000

# Number of threads for image preprocessing. Should be a multiple of 2.
num_preprocess_threads = 4

# Image embedding size
embedding_size = 512 # 100

# Number of units in LSTM
num_units = 1000 # 100

# If < 1.0, the dropout keep probability applied to LSTM variables.
lstm_dropout_keep_prob = 0.7

# Learning rate for the initial phase of training.
initial_learning_rate = 2.0
learning_rate_decay_factor = 0.5
num_epochs_per_decay = 8.0

# Number of examples per epoch of training data.
num_examples_per_epoch = 586363

# Optimizer for training the model.
optimizer = "SGD"

# Learning rate when fine tuning the Inception v3 parameters.
train_inception_learning_rate = 0.0005

# If not None, clip gradients to this value.
clip_gradients = 5.0

# How many model checkpoints to keep.
max_checkpoints_to_keep = 5

# Batch size.
batch_size = 32
