#!/usr/bin/env python
# coding: utf-8

# # DCGAN for generating MNIST-like data
# #### Original DCGAN paper: https://arxiv.org/pdf/1511.06434
# #### Inspired by: https://www.wouterbulten.nl/blog/tech/getting-started-with-generative-adversarial-networks/
# #### Zach D.

# In[ ]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np


# ## Defining the Network
# ### Generator

# In[9]:


def generator():
    net = tf.keras.Sequential()
    
    net.add(tf.keras.layers.Dense(4*4*1024,
                                    input_shape=(100,)))
    net.add(tf.keras.layers.Reshape(target_shape=(4,4,1024)))
    net.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    net.add(tf.keras.layers.Activation('relu'))
              
    net.add(tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(5,5),
                                              strides=(2,2), padding='same'))
    net.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    net.add(tf.keras.layers.Activation('relu'))
              
    net.add(tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(5,5),
                                              strides=(2,2), padding='same'))
    net.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    net.add(tf.keras.layers.Activation('relu'))
    
    net.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5,5),
                                              strides=(2,2), padding='same'))
    net.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    net.add(tf.keras.layers.Activation('relu'))
    
    net.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(5,5),
                                              strides=(2,2), padding='same'))
    net.add(tf.keras.layers.Activation('tanh'))
    
    return net
    


# ### Discriminator

# In[10]:


def discriminator():
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Conv2D(filters=64, 
                                     strides=(2, 2),
                                     kernel_size=(5, 5),
                                     input_shape=(64,64,1),
                                     padding='same'))
    net.add(tf.keras.layers.LeakyReLU(0.2))
    
    net.add(tf.keras.layers.Conv2D(filters=128, 
                                     strides=(2, 2),
                                     kernel_size=(5, 5),
                                     padding='same'))
    net.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    net.add(tf.keras.layers.LeakyReLU(0.2))
    
    net.add(tf.keras.layers.Conv2D(filters=256, 
                                     strides=(2, 2),
                                     kernel_size=(5, 5),
                                     padding='same'))
    net.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    net.add(tf.keras.layers.LeakyReLU(0.2))
    
    net.add(tf.keras.layers.Conv2D(filters=512, 
                                     strides=(2, 2),
                                     kernel_size=(5, 5),
                                     padding='same'))
    net.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    net.add(tf.keras.layers.LeakyReLU(0.2))
    
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(1))
    net.add(tf.keras.layers.Activation('sigmoid'))
    
    return net


# ### Adversarial Network (G + D)

# In[33]:


net_discriminator=discriminator()
net_generator=generator()

optim_discriminator = tf.keras.optimizers.RMSprop(lr=0.00008, decay=1e-10)
model_discriminator = tf.keras.Sequential()
model_discriminator.add(net_discriminator)
model_discriminator.compile(loss='binary_crossentropy', optimizer=optim_discriminator, metrics=['accuracy'])

optim_adversarial = tf.keras.optimizers.Adam(lr=0.0004, decay=1e-10)
model_adversarial = tf.keras.Sequential()
model_adversarial.add(net_generator)

for layer in net_discriminator.layers:
    layer.trainable = False

model_adversarial.add(net_discriminator)
model_adversarial.compile(loss='binary_crossentropy', optimizer=optim_adversarial, metrics=['accuracy'])
model_adversarial.summary()


# ## Visualize and load the real data

# In[19]:


(x_train, y_train), (_,_) = tf.keras.datasets.mnist.load_data()
print(x_train[1].shape)
batch_size = 64


# In[20]:


def preprocess_fn(image, label):
    x = tf.reshape(tf.cast(image, tf.float32), (28,28,1))
    x = tf.image.resize_images(x, (64,64))
    x = tf.reshape(tf.cast(x, tf.float32), (64,64,1))
    x /= 255.0
    y = tf.one_hot(label, 10)
    return x,y
mnist_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
mnist_ds = mnist_ds.shuffle(10000)
mnist_ds = mnist_ds.apply(tf.data.experimental.map_and_batch(
        preprocess_fn, batch_size=batch_size, num_parallel_batches=6, 
        drop_remainder=True))
# real_ds = real_ds.map(real_ds_preprocess_fn)
mnist_ds = mnist_ds.repeat()
mnist_ds = mnist_ds.prefetch(tf.data.experimental.AUTOTUNE)


# In[60]:


batch_size = 64


# ## Create fake data and train network

# In[ ]:


vis_noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
import random
loss_adv = []
loss_dis = []
acc_adv = []
acc_dis = []
plot_iteration = []
import matplotlib.pyplot as plt

mnist_iterator = mnist_ds.make_one_shot_iterator() 
for i in range(500):
    real_images, real_labels = mnist_iterator.get_next() # get a batch from real_ds
    real_labels = np.ones([batch_size,])

    noise = np.random.uniform(-1, 1.0, size=[batch_size, 100])
    fake_images = net_generator.predict(noise) # create fake batch of images
    fake_labels = np.zeros([batch_size,]) # fake images are labeled as 0
    if i % 25 == 0:
        rand_n = random.randint(0,batch_size-1)
        fake_images_plt = np.squeeze(fake_images,axis=3)
        plt.gray()
        plt.imshow(fake_images_plt[rand_n])
        plt.show()
        plt.savefig('gen_img_'+str(i)+'.jpg')

    # Train D
    # Fake data
    model_discriminator.train_on_batch(fake_images, fake_labels)
    # Real Data
    model_discriminator.train_on_batch(real_images, real_labels)
    
#     # Train G
    y = tf.ones((batch_size), dtype='uint8')
    noise = np.random.uniform(-1, 1.0, size=[batch_size, 100])
    a_stats = model_adversarial.train_on_batch(noise, y)


# In[58]:


# Debugging stuff

mnist_iterator = mnist_ds.make_one_shot_iterator()
real_images, real_labels = mnist_iterator.get_next()
real_labels = np.ones([batch_size])
print(real_labels[0:3])
predictions = model_discriminator(real_images, training=False)

noise = tf.random.uniform(shape=(batch_size,100,), minval=0, maxval= 1.0)
import matplotlib.pyplot as plt



fake_images = net_generator(noise, training=False)
print(fake_images[0,:,:,:].shape)

predictions2 = model_discriminator(fake_images, training=False)
fake_images = np.squeeze(fake_images,axis=2)
plt.gray()
plt.imshow(fake_images[0])
plt.show()
print(predictions)
print(predictions2)


# In[ ]:





# In[ ]:





# In[ ]:




