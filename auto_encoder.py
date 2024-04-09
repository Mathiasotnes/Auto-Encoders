from stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape, Input
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import numpy as np

class AutoEncoderNet:
    
    def __init__(self, force_learn: bool = False, file_name: str = "./models/autoencoder/autoencoder", encoding_dim: np.int64 = 128, input_shape: tuple = (28, 28, 1)):
        self.force_relearn = force_learn
        self.file_name = file_name
        self.encoding_dim = encoding_dim
        self.input_shape = input_shape
        self.encoder, self.decoder, self.autoencoder = self._build_model()

    def _build_model(self):
        # Encoder
        encoder = Sequential(name='encoder')
        encoder.add(Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same', input_shape=self.input_shape))
        encoder.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same'))
        encoder.add(Flatten())
        encoder.add(Dense(self.encoding_dim, activation='relu'))
        
        # Decoder
        # The encoder CNN layers have reduced the image size to 7x7 before flattening and encoding
        # So we're doing the reverse operation in our decoder
        decoder = Sequential(name='decoder')
        decoder.add(Dense(7*7*64, activation='relu', input_shape=(self.encoding_dim,)))
        decoder.add(Reshape((7, 7, 64)))
        decoder.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        decoder.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        decoder.add(Conv2DTranspose(self.input_shape[-1], (3, 3), strides=(1, 1), padding='same', activation='sigmoid'))

        # Input for full autoencoder
        autoencoder_input = Input(shape=self.input_shape)
        # Connect encoder and decoder
        encoded = encoder(autoencoder_input)
        decoded = decoder(encoded)
        # Full autoencoder model
        autoencoder = Model(autoencoder_input, decoded, name="autoencoder")
        autoencoder.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
        
        return encoder, decoder, autoencoder

    def train(self, generator: StackedMNISTData, epochs: np.int64 = 10):
        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """
        self.done_training = self.load_weights()

        if self.force_relearn or self.done_training is False:
            # Get hold of data
            x_train, y_train = generator.get_full_data_set(training=True)
            x_test, y_test = generator.get_full_data_set(training=False)

            # "Translate": Only look at "red" channel; only use the last digit. Use one-hot for labels during training
            x_train = x_train[:, :, :, [0]]
            y_train = keras.utils.to_categorical((y_train % 10).astype(np.int64), 10)
            x_test = x_test[:, :, :, [0]]
            y_test = keras.utils.to_categorical((y_test % 10).astype(np.int64), 10)

            # Fit model
            self.autoencoder.fit(x=x_train, y=x_train, batch_size=1024, epochs=epochs,
                           validation_data=(x_test, x_test))

            # Save weights and leave
            self.autoencoder.save_weights(filepath=self.file_name)
            self.done_training = True

        return self.done_training

    def load_weights(self):
        # noinspection PyBroadException
        try:
            self.autoencoder.load_weights(filepath=self.file_name)
            # print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print(f"Could not read weights for verification_net from file. Must retrain...")
            done_training = False

        return done_training

    def generate_new_images(self, num_images=10):
        """
        Generate new images from random noise
        """
        z = 2*np.random.randn(num_images, self.encoding_dim)
        generated_images = self.decoder(z)
        return generated_images
    
    def show_reconstructions(self, generator: StackedMNISTData, num_images=10):
        """
        Compare original images with their reconstructions
        """
        x_test, _ = generator.get_random_batch(training=False, batch_size=num_images)
        x_test = x_test[:, :, :, [0]]
        decoded_imgs = self.autoencoder.predict(x_test)

        plt.figure(figsize=(16, 4))
        for i in range(num_images):
            # Original
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.set_title('Original')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Reconstruction
            ax = plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.set_title('Reconstructed')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    
    def show_generated_images(self, num_images=10):
        """
        Show generated images
        """
        generated_images = self.generate_new_images(num_images=num_images)
        plt.figure(figsize=(16, 4))
        plt.suptitle('Randomly generated images')
        for i in range(num_images):
            ax = plt.subplot(1, num_images, i + 1)
            plt.imshow(generated_images[i].numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    
    def calculate_reconstruction_error(self, images):
        """
        Calculate the mean squared error reconstruction loss for each image
        """
        reconstructed_images = self.autoencoder.predict(images)
        mse = np.mean(np.square(images - reconstructed_images), axis=(1, 2, 3))
        return mse

    def display_anomalies(self, generator: StackedMNISTData, num_images=10):
        """
        Display images with the highest reconstruction error
        """
        x_test, _ = generator.get_full_data_set(training=False)
        x_test = x_test[:, :, :, [0]]
        errors = self.calculate_reconstruction_error(x_test)
        
        # Find the images with the highest reconstruction error
        idxs = np.argsort(errors)[-num_images:]
        anomalous_images = x_test[idxs]
        
        plt.figure(figsize=(16, 4))
        plt.suptitle('Images with the highest reconstruction error')
        for i, idx in enumerate(idxs):
            ax = plt.subplot(1, num_images, i + 1)
            plt.imshow(anomalous_images[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    net = AutoEncoderNet(encoding_dim=128, file_name="./models/autoencoder/autoencoder")
    net.train(generator=gen, epochs=10)

    # Reconstruct some images
    net.show_reconstructions(generator=gen, num_images=10)

    # Generate new images
    net.show_generated_images(num_images=10)

    # Display images with the highest reconstruction error
    net.display_anomalies(generator=gen, num_images=10)


"""
====================================================
Conclusion about the performance of the auto encoder
====================================================

--------------
Reconstruction
--------------
The auto encoder can reconstruct images with high accuracy. The reconstructed images are very similar to the original images.

----------
Generation
----------
The auto encoder can generate new images from random noise, but the generated images look nothing like hand-writing. This
is because we don't know the representation of the encoded images, and choosing random noise for the encoded images doesn't
produce meaningful results.

-----------------
Anomaly Detection
-----------------
The auto encoder can detect anomalies in the dataset by calculating the reconstruction error for each image. The images with
the highest reconstruction error are considered anomalies. The auto encoder can detect anomalies with high accuracy.

"""