import os

from keras.layers import Reshape, InputLayer, Flatten, LeakyReLU, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras_adversarial import (AdversarialOptimizerSimultaneous, 
                               normal_latent_sampling, simple_gan,
                               gan_targets, AdversarialModel)
from keras_adversarial.legacy import Dense, l1l2
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

def main():
    data_dir = "goldens_32x32/"
    out_dir = "m_gan_out/"
    epochs = 1
    batch_size = 16 

    # TODO: Research why these values were chosen
    opt_g = Adam(1e-4, decay=1e-5)
    opt_d = Adam(1e-3, decay=1e-5)
    loss='binary_crossentropy'
    latent_dim = IMAGE_DIM
    adversarial_optimizer = AdversarialOptimizerSimultaneous()

    generator = get_generator()
    discriminator = get_discriminator()
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
        img = imread(image_path, flatten=True)
        img = img.astype('float32')
        temp.append(img)
        
    train_x = np.stack(temp)
    train_x = train_x / 255
    
    # Side effects
    model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=epochs, batch_size=batch_size)    
    
    # TODO: Investigate if this is the source of all-white image output
    zsamples = np.random.normal(size=(10, latent_dim))
    # pred = generator.predict(zsamples).reshape((10, 10, IMAGE_DIM, IMAGE_DIM))
    pred = generator.predict(zsamples)
    for i in range(pred.shape[0]):
        plt.imshow(pred[i, :])
        plt.savefig(out_dir+str(i)+'.png')

if __name__ == "__main__":
    main()
