import os

from keras.layers import Reshape, InputLayer, Flatten, LeakyReLU, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras_adversarial import (AdversarialOptimizerSimultaneous, 
                               normal_latent_sampling, simple_gan,
                               gan_targets, AdversarialModel)
from keras_adversarial.legacy import Dense, l1l2, Convolution2D, BatchNormalization, AveragePooling2D
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

IMAGE_DIM = 32 

def get_generator():
    """
    Returns the generator model
    """
    reg = lambda: l1l2(1e-5, 1e-5)
    model = Sequential()
    model.add(Dense(256, input_dim=IMAGE_DIM, W_regularizer=reg()))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512, W_regularizer=reg()))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1024, W_regularizer=reg()))
    model.add(LeakyReLU(0.2))

    # I think this has to be done to make the discriminator work
    model.add(Dense(np.prod((IMAGE_DIM, IMAGE_DIM)), W_regularizer=reg()))
    model.add(Activation('sigmoid'))
    model.add(Reshape((IMAGE_DIM, IMAGE_DIM)))
    return model

def get_discriminator():
    """
    Returns the discriminator model
    """
    reg = lambda: l1l2(1e-5, 1e-5)
    model = Sequential()
    model.add(Flatten(name="discriminator_flatten", input_shape=(IMAGE_DIM, IMAGE_DIM)))
    model.add(Dense(1024, W_regularizer=reg()))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512, W_regularizer=reg()))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256, W_regularizer=reg()))
    model.add(Dense(1, W_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model

def get_generator_cifar():
    model = Sequential()
    nch = 256
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)
    h = 5
    model.add(Dense(nch * 4 * 4, input_dim=100, W_regularizer=reg()))
    model.add(BatchNormalization(mode=0))
    model.add(Reshape((4, 4, nch)))
    model.add(Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 4), h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, h, h, border_mode='same', W_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model


def get_discriminator_cifar():
    nch = 256
    h = 5
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)

    c1 = Convolution2D(int(nch / 4), h, h, border_mode='same', W_regularizer=reg(),
                       input_shape=(32, 32, 3))
    c2 = Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg())
    c3 = Convolution2D(nch, h, h, border_mode='same', W_regularizer=reg())
    c4 = Convolution2D(1, h, h, border_mode='same', W_regularizer=reg())

    model = Sequential()
    model.add(c1)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(c2)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(c3)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(c4)
    model.add(AveragePooling2D(pool_size=(4, 4), border_mode='valid'))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model


def main():
    data_dir = "goldens_filtered_32x32_gray/"
    out_dir = "m_gan_out/"
    epochs = 1 
    batch_size = 64 

    # TODO: Research why these values were chosen
    opt_g = Adam(1e-4, decay=1e-5)
    opt_d = Adam(1e-3, decay=1e-5)
    loss='binary_crossentropy'
    latent_dim = 100 
    adversarial_optimizer = AdversarialOptimizerSimultaneous()
    
    # My simple models
    # generator = get_generator()
    # discriminator = get_discriminator()

    # CIFAR example convolutional models
    generator = get_generator_cifar()
    discriminator = get_discriminator_cifar()

    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))

    # print summary of models
    generator.summary()
    discriminator.summary()
    gan.summary()

    # build adversarial model
    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[opt_g, opt_d],
                              loss=loss)
    
    temp = []
    for img_name in os.listdir(data_dir):
        image_path = data_dir + img_name
        img = imread(image_path)
        img = img.astype('float32')
        temp.append(img)
        
    train_x = np.stack(temp)
    train_x = train_x / 255
    
    # Side effects
    model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=epochs, batch_size=batch_size)    
    
    zsamples = np.random.normal(size=(10, latent_dim))
    pred = generator.predict(zsamples)
    for i in range(pred.shape[0]):
        plt.imshow(pred[i, :])
        plt.savefig(out_dir+str(i)+'.png')

if __name__ == "__main__":
    main()
