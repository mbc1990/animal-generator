import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras.layers import Reshape, Flatten, LeakyReLU, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import normal_latent_sampling, AdversarialOptimizerSimultaneous
from keras_adversarial.legacy import l1l2, Dense, fit
from keras.layers import Reshape, Flatten, LeakyReLU, Activation
from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras_adversarial.image_grid_callback import ImageGridCallback

from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras_adversarial.legacy import Dense, BatchNormalization, fit, l1l2, Convolution2D, AveragePooling2D
import keras.backend as K
from image_utils import dim_ordering_fix, dim_ordering_unfix, dim_ordering_shape

# These are the models from the keras-adversarial CIFAR example
def model_generator_cifar():
    model = Sequential()
    nch = 256
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)
    h = 5
    model.add(Dense(nch * 4 * 4, input_dim=100, W_regularizer=reg()))
    model.add(BatchNormalization(mode=0))
    model.add(Reshape(dim_ordering_shape((nch, 4, 4))))
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


def model_discriminator_cifar():
    nch = 256
    h = 5
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)
    '''
    c1 = Convolution2D(int(nch / 4), h, h, border_mode='same', W_regularizer=reg(),
                       input_shape=dim_ordering_shape((3, 32, 32)))
    '''

    # M: I've modified the input shape to (8, 256, 256) representing
    # M: the 8 bit grayscale images with 256x256 resolution
    # M: (Or 32x32 for debugging)
    input_shape = dim_ordering_shape((3, 32, 32))
    c1 = Convolution2D(int(nch / 4), h, h, border_mode='same', W_regularizer=reg(),
                       input_shape=dim_ordering_shape((32, 32, 3)))
    print "c1..."
    c2 = Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg())
    print "c2..."
    c3 = Convolution2D(nch, h, h, border_mode='same', W_regularizer=reg())
    print "c3..."
    c4 = Convolution2D(1, h, h, border_mode='same', W_regularizer=reg())
    print "c4..."

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
    # set path
    root_dir = os.path.abspath('.')
    data_dir = os.path.join(root_dir, 'MData')
            
    # load data
    train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
    # test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    temp = []
    for img_name in train.filename:
        image_path = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)
        img = imread(image_path, flatten=True)
        img = img.astype('float32')
        temp.append(img)
        
    train_x = np.stack(temp)
    train_x = train_x / 255

    epochs = 1 
    batch_size = 128    

    model_1 = model_generator_cifar()
    model_2 = model_discriminator_cifar()

    # gan = simple_gan(model_1, model_2, normal_latent_sampling((100,)))
    latent_dim = 100
    gan = simple_gan(model_1, model_2, latent_sampling=normal_latent_sampling((latent_dim,)))

    model = AdversarialModel(base_model=gan,player_params=[model_1.trainable_weights, model_2.trainable_weights])
    model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=['adam', 'adam'], loss='binary_crossentropy')
    
    history = model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=epochs, batch_size=batch_size)    
    zsamples = np.random.normal(size=(10, 100))
    pred = model_1.predict(zsamples)
    for i in range(pred.shape[0]):
        plt.imshow(pred[i, :], cmap='gray')
        plt.savefig('out/animals/'+str(i)+'.png')


if __name__ == "__main__":
    main()
