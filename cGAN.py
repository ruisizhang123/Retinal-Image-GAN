# Author: Ruisi Zhang

import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def Discriminator(image_shape,vessal_shape):
        Discriminator = Sequential()

        Discriminator.add(Conv2D(filters=64, kernel_size=3,
                                 strides=2, padding="same",
                                 input_shape=(image_shape)))
        Discriminator.add(LeakyReLU(alpha=0.2))

        Discriminator.add(BatchNormalization(momentum=0.8))
        Discriminator.add(Conv2D(filters=128, kernel_size=3,
                                 strides=2, padding="same"))
        Discriminator.add(LeakyReLU(alpha=0.2))

        Discriminator.add(BatchNormalization(momentum=0.8))
        Discriminator.add(Conv2D(filters=256, kernel_size=3, 
                                 strides=2, padding="same"))
        Discriminator.add(LeakyReLU(alpha=0.2))

        Discriminator.add(BatchNormalization(momentum=0.8))
        Discriminator.add(Conv2D(filters=512, kernel_size=3, 
                                 strides=1, padding="same"))
        Discriminator.add(LeakyReLU(alpha=0.2))

        Discriminator.add(Flatten())
        Discriminator.add(Dense(1, activation='sigmoid'))

        img = Input(image_shape)
        label = Input(vessal_shape)
        labels = Dense(image_shape[0]*image_shape[1]*image_shape[2], activation='relu')(label)
        labels = Reshape((vessal_shape[0], vessal_shape[1], vessal_shape[2]))(labels)
        input_img_label = multiply([img, labels])
        validity = model(input_img_label)

        return discriminator([img, label], validity)


def Generator(image_shape, vessal_shape):
    Generator = Sequential()

    Generator.add(Dense(filters=512 * vessal_shape[0]/8 * vessal_shape[1]/8, input_shape=noise_shape))
    Generator.add(Reshape((vessal_shape[0]/8, vessal_shape[1]/8, 512)))
    Generator.add(Activation("relu"))

    Generator.add(BatchNormalization(momentum=0.8))
    Generator.add(UpSampling2D())
    Generator.add(Conv2D(filters=256, kernel_size=3, padding="same"))
    Generator.add(Activation("relu"))

    Generator.add(BatchNormalization(momentum=0.8))
    Generator.add(UpSampling2D())
    Generator.add(Conv2D(filters=128, kernel_size=3, padding="same"))
    Generator.add(Activation("relu"))

    Generator.add(BatchNormalization(momentum=0.8))
    Generator.add(UpSampling2D())
    Generator.add(Conv2D(filters=64, kernel_size=3, padding="same"))
    Generator.add(Activation("relu"))

    Generator.add(Conv2D(self.channel, kernel_size=3, padding="same"))
    Generator.add(Activation("tanh"))


    noise = Input(noise_shape)
    label = Input(vessal_shape)
    labels = Dense(self.z_dim, activation='relu')(label)
    input_noise_label = multiply([noise, labels])
    validity = Generator(input_noise_label)

    return generator([noise, label], validity)

def Train(batch_size, epochs, image_shape, vessal_shape, dataset_path, dataset_vessal_path):
    dataset_generator = ImageDataGenerator()
    dataset_generator = dataset_generator.flow_from_directory(
        dataset_path, target_size=(image_shape[0], image_shape[1]),
        batch_size=batch_size,
        class_mode=None)
    dataset_vessal_generator = ImageDataGenerator()
    dataset_vessal_generator = dataset_generator.flow_from_directory(
        dataset_vessal_path, target_size=(image_shape[0], image_shape[1]),
        batch_size=batch_size,
        class_mode=None)

    generator = Generator(image_shape,vessal_shape)
    discriminator = Discriminator(image_shape,vessal_shape)

    gan = Sequential()
    
    ## Combine Networks
    discriminator.trainable = False
    gan.add(generator)
    gan.add(discriminator)

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer,
                metrics=None)
    

    number_of_batches = int(21352 / batch_size)

    adversarial_loss = np.empty(shape=1)
    discriminator_loss = np.empty(shape=1)
    batches = np.empty(shape=1)

    plt.ion()

    current_batch = 0

    for epoch in range(epochs):

        print("Epoch " + str(epoch+1) + "/" + str(epochs) + " :")

        for batch_number in range(number_of_batches):

            start_time = time.time()

            real_images = dataset_generator.next()
            real_images /= 127.5
            real_images -= 1

            current_batch_size = real_images.shape[0]

            noise = np.random.normal(0, 1,
                                     size=(current_batch_size,) + (1, 1, 100))

            generated_images = generator.predict(noise)

            real_y = (np.ones(current_batch_size) -
                      np.random.random_sample(current_batch_size) * 0.2)
            fake_y = np.random.random_sample(current_batch_size) * 0.2

            discriminator.trainable = True

            d_loss = discriminator.train_on_batch(real_images, real_y)
            d_loss += discriminator.train_on_batch(generated_images, fake_y)

            discriminator_loss = np.append(discriminator_loss, d_loss)

            discriminator.trainable = False

            noise = np.random.normal(0, 1,
                                     size=(current_batch_size * 2,) +
                                     (1, 1, 100))

            fake_y = (np.ones(current_batch_size * 2) -
                      np.random.random_sample(current_batch_size * 2) * 0.2)

            g_loss = gan.train_on_batch(noise, fake_y)
            adversarial_loss = np.append(adversarial_loss, g_loss)
            batches = np.append(batches, current_batch)

            if((batch_number + 1) % 50 == 0 and
               current_batch_size == batch_size):
                SaveImage(generated_images, epoch, batch_number)

            time_elapsed = time.time() - start_time

            print(" Batch " + str(batch_number + 1) + "/" +
                  str(number_of_batches) +
                  " generator loss | discriminator loss : " +
                  str(g_loss) + " | " + str(d_loss) + ' - batch took ' +
                  str(time_elapsed) + ' s.')

            current_batch += 1

        if (epoch + 1) % 5 == 0:
            discriminator.trainable = True
            generator.save('models/generator_epoch' + str(epoch) + '.hdf5')
            discriminator.save('models/discriminator_epoch' +
                               str(epoch) + '.hdf5')

        plt.figure(1)
        plt.plot(batches, adversarial_loss, color='green',
                 label='Generator Loss')
        plt.plot(batches, discriminator_loss, color='blue',
                 label='Discriminator Loss')
        plt.title("DCGAN Train")
        plt.xlabel("Batch Iteration")
        plt.ylabel("Loss")
        if epoch == 0:
            plt.legend()
        plt.pause(0.0000000001)
        plt.show()
        plt.savefig('trainingLossPlot.png')

## Save Genertaed Images
def SaveImage(generated_images, epoch, batch_number):

    plt.figure(figsize=(8, 8), num=2)
    gs1 = gridspec.GridSpec(8, 8)
    gs1.update(wspace=0, hspace=0)

    for i in range(64):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        fig = plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    save_name = 'generated images/generatedSamples_epoch' + str(
        epoch + 1) + '_batch' + str(batch_number + 1) + '.png'

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.pause(0.0000000001)
    plt.show()

def main():
    dataset_path = '/mac/cGAN/data1/'
    dataset_vessal_path = '/mac/cGAN/data2/'
    batch_size = 64
    image_shape = (32, 32, 3)
    vessal_shape = (32, 32, 3)
    epochs = 300
    Train(batch_size, epochs,
                image_shape, dataset_path)

if __name__ == "__main__":
    main()