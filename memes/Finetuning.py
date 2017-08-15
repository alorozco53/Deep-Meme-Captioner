
# coding: utf-8

# # Convolutional net finetuning

# In[1]:


import numpy as np
import os

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from finetune import load_data, to_categorical


# ## Settings

# In[3]:


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)


# In[4]:


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1000, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)


# In[5]:


# this is the model we will train
model = Model(base_model.input, predictions)


# In[6]:


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = True


# In[7]:


# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# ## Get dataset

# In[8]:


meme_path = '/media/backup/crawling/memes/meme_characters/'
non_meme_path = '/media/backup/crawling/memes/test2014/'
m_train, m_test, n_train, n_test = load_data(meme_path, non_meme_path)


# In[9]:


# Training set
x_train = np.vstack([m_train, n_train])
my_train = np.ones((m_train.shape[0], 1))
ny_train = np.zeros((n_train.shape[0], 1))
y_train = to_categorical(np.vstack([my_train, ny_train]), num_classes=2)


# In[10]:


# Validation set
x_test = np.vstack([m_test, n_test])
my_test = np.ones((m_test.shape[0], 1))
ny_test = np.zeros((n_test.shape[0], 1))
y_test = to_categorical(np.vstack([my_test, ny_test]), num_classes=2)


# ## Testing while training

# In[11]:


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


# ## Callbacks

# In[12]:


logdir = 'inception_log2.0/'


# In[13]:


tb = TensorBoard(log_dir=logdir, histogram_freq=1, write_images=True)


# ## Training

# In[14]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# In[16]:


model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=32),
                    samples_per_epoch = len(x_train) / 32,
                    nb_epoch=80,
                    callbacks=[tb],
                    # initial_epoch=71,
                    validation_data=test_datagen.flow(x_test, y_test, batch_size=32),
                    nb_val_samples=1600 / 32)
model.save(os.path.join(logdir, 'fine_inception.h5'))


# In[ ]:


score = model.evaluate_generator(test_datagen.flow(x_test, y_test, batch_size=32), 32)
print('score:', score)
