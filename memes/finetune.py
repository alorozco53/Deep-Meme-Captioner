#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import keras
import os
import argparse

from keras.callbacks import TensorBoard
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def build_model(input_shape=(299, 299, 3), output_dim=2):
    img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', name='conv1')(img)
    x = Conv2D(16, (3, 3), activation='relu', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pool1')(x)
    x = Dropout(0.25, name='droput1')(x)

    # x = Conv2D(64, (3, 3), activation='relu', name='conv3')(x)
    # x = Conv2D(64, (3, 3), activation='relu', name='conv4')(x)
    # x = MaxPooling2D(pool_size=(2, 2), name='max_pool2')(x)
    # x = Dropout(0.25, name='droput2')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout3')(x)
    x = Dense(output_dim, activation='softmax', name='fc2')(x)
    return Model(img, x)

def train(x_train,
          y_train,
          model,
          datagen,
          test_datagen,
          logdir,
          validation_steps=250,
          batch_size=32,
          epochs=10,
          initial_epoch=0):
    # callback config
    tb = TensorBoard(log_dir=logdir, write_graph=True, histogram_freq=0, write_images=True)

    # fits the model on batches with real-time data augmentation:
    return model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                               steps_per_epoch = len(x_train) / batch_size,
                               epochs=epochs + initial_epoch,
                               callbacks=[tb],
                               initial_epoch=initial_epoch,
                               validation_data=test_datagen,
                               validation_steps=validation_steps)

def load_data(meme_path, non_meme_path, size=3000):
    assert os.path.exists(meme_path)
    assert os.path.exists(non_meme_path)

    # Load memes
    print('Loading memes from:', meme_path)
    count = 0
    memes = []
    memest = []
    for meme in os.listdir(meme_path):
        img_path = os.path.join(meme_path, meme, '{}.jpg'.format(meme))
        if os.path.exists(img_path):
            img = image.load_img(img_path, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            count += 1
            if count >= size:
                memest.append(x)
            else:
                memes.append(x)
    total_size = len(memes) + len(memest)
    memes = np.array(memes).squeeze()
    memest = np.array(memest).squeeze()

    # Load non-memes
    print('Loading non-memes from:', non_meme_path)
    count = 0
    non_memes = []
    non_memest = []
    for non_meme in os.listdir(non_meme_path):
        img_path = os.path.join(non_meme_path, non_meme)
        if os.path.exists(img_path):
            img = image.load_img(img_path, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            count += 1
            if count >= total_size:
                break
            elif count >= size:
                non_memest.append(x)
            else:
                non_memes.append(x)

    non_memes = np.array(non_memes).squeeze()
    non_memest = np.array(non_memest).squeeze()

    print('Dataset successfully loaded!')
    return memes, memest, non_memes, non_memest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script\'s argument parser')
    parser.add_argument('train_dir', help='directory where training checkpoints will be stored')
    parser.add_argument('meme_path', help='directory where memes are stored')
    parser.add_argument('non_meme_path', help='directory where non-memes are stored')
    parser.add_argument('--data_size', default=3000, help='training data size', type=int)
    parser.add_argument('--batch_size', default=32, help='training batch size', type=int)
    parser.add_argument('--epochs', default=10, help='number of epochs to be trained', type=int)
    parser.add_argument('--initial_epoch', default=0, help='initial training epoch', type=int)
    args = parser.parse_args()

    filepath = args.train_dir
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    # Load data
    m_train, m_test, n_train, n_test = load_data(args.meme_path,
                                                 args.non_meme_path,
                                                 args.data_size)

    # Train data
    x_train = np.vstack([m_train, n_train])
    my_train = np.ones((m_train.shape[0], 1))
    ny_train = np.zeros((n_train.shape[0], 1))
    y_train = to_categorical(np.vstack([my_train, ny_train]), num_classes=2)

    # Test data
    x_test = np.vstack([m_test, n_test])
    my_test = np.ones((m_test.shape[0], 1))
    ny_test = np.zeros((n_test.shape[0], 1))
    y_test = to_categorical(np.vstack([my_test, ny_test]), num_classes=2)

    # Train data generator
    datagen = ImageDataGenerator(
        featurewise_center=True, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False) # randomly flip images
    print('Fitting data generator...')
    datagen.fit(x_train)

    # Load model
    try:
        model_path = os.path.join(filepath, 'model.h5')
        model = load_model(model_path)
        print('model loaded from:', model_path)
    except:
        # model = build_model(input_shape=(150, 150, 3))
        base_model = InceptionV3(weights='imagenet',
                                 include_top=False,
                                 input_shape=(150, 150, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)
        model = Model(base_model.input, predictions)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      metrics=['mae', 'acc'],
                      optimizer=sgd)
        print('model built from scratch')
        print(model.summary())

    # Test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    print('Fitting data generator...')
    test_datagen.fit(x_test)

    for i in range(2):
        # Training
        if i == 0:
            ie = args.initial_epoch
            eps = 50
        else:
            ie = eps
            eps = args.epochs
            # we chose to train the top 2 inception blocks, i.e. we will freeze
            # the first 249 layers and unfreeze the rest:
            for layer in model.layers[:249]:
                layer.trainable = False
            for layer in model.layers[249:]:
                layer.trainable = True
            model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metrics=['mae', 'acc'])
        try:
            train(x_train, y_train, model,
                  datagen,
                  test_datagen.flow(x_test, y_test, batch_size=args.batch_size),
                  filepath,
                  batch_size=args.batch_size,
                  epochs=eps,
                  initial_epoch=ie)
            score = model.evaluate_generator(datagen.flow(x_test, y_test, batch_size=32), 32)
            print('score:', score)
        except:
            print('Training stopped!')

    # save model
    model_path = os.path.join(filepath, 'model.h5')
    model.save(model_path)
    print('model saved in:', model_path)
