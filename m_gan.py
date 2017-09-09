import os

from keras.layers import Reshape, InputLayer, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras_adversarial import (AdversarialOptimizerSimultaneous, 
                               normal_latent_sampling, simple_gan,
                               gan_targets, AdversarialModel)
from keras_adversarial.legacy import Dense
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

# d_input_shape = (150, 150, 3)

def get_generator():
    """
    Returns the generator model
    """
    model = Sequential()
    model.add(Dense(500, input_dim=150))
    model.add(Dense(units=150*150, activation='sigmoid'))
    model.add(Reshape((150, 150)))
    return model

def get_discriminator():
    """
    Returns the discriminator model
    """
    model = Sequential()
    model.add(Flatten(name="discriminator_flatten", input_shape=(150, 150)))
    model.add(InputLayer(input_shape=(150, 150)))
    model.add(Dense(500))
    model.add(Dense(units=1, activation='sigmoid'))
    return model

def main():
    data_dir = "cats_150x150/"
    out_dir = "m_gan_out/"
    img_w = 150
    img_h = 150
    epochs = 2
    batch_size = 16 

    # TODO: Research why these values were chosen
    opt_g = Adam(1e-4, decay=1e-5)
    opt_d = Adam(1e-3, decay=1e-5)
    loss='binary_crossentropy'
    latent_dim = 150
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

    history = model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=epochs, batch_size=batch_size)    
    zsamples = np.random.normal(size=(10, latent_dim))
    pred = generator.predict(zsamples)
    for i in range(pred.shape[0]):
        plt.savefig(out_dir+str(i)+'.png')

if __name__ == "__main__":
    main()
