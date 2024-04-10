from stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import numpy as np

class SamplingLayer(keras.layers.Layer):
    """Uses (mean, log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

class VarAutoEncoder(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    def call(self, inputs, training=False):
        mean, log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed
 
    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
 
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class VarAutoEncoderNet:
    
    def __init__(self, mode: str = 'mono', force_learn: bool = False, file_name: str = "./models/vae/vae", encoding_dim: np.int64 = 24):
        self.mode = mode
        self.force_relearn = force_learn
        self.file_name = file_name
        self.encoding_dim = encoding_dim
        if self.mode == 'mono':
            self.input_shape = (28, 28, 1)
        elif self.mode == 'color':
            self.input_shape = (28, 28, 3)
        else:
            raise ValueError("Mode must be 'mono' or 'color'")
        self.encoder, self.decoder, self.vae = self._build_model()
    
    def _build_encoder(self):
        encoder_inputs = keras.Input(shape=self.input_shape)
        x = Conv2D(64, (3, 3), activation="relu", strides=(2, 2), padding="same")(encoder_inputs)
        x = Conv2D(128, (3, 3), activation="relu", strides=(2, 2), padding="same")(x)
        x = Flatten()(x)
        x = Dense(16, activation="relu")(x)

        # X is connected to two outputs: mean and log_var
        mean = Dense(self.encoding_dim, name="mean")(x)
        log_var = Dense(self.encoding_dim, name="log_var")(x)

        # mean and log_var layers are connected to the sampling layer, which samples z given mean and log_var
        z = SamplingLayer()([mean, log_var])

        # Need mean and log_var for KL divergence loss, and z is used as input to the decoder
        encoder = keras.Model(encoder_inputs, [mean, log_var, z], name="encoder")
        encoder.summary()
        return encoder

    def _build_decoder(self):
        z = keras.Input(shape=(self.encoding_dim,))
        x = Dense(7 * 7 * 64, activation="relu")(z)
        x = Reshape((7, 7, 64))(x)
        x = Conv2DTranspose(128, (3, 3), activation="relu", strides=(2, 2), padding="same")(x)
        x = Conv2DTranspose(64, (3, 3), activation="relu", strides=(2, 2), padding="same")(x)
        decoder_outputs = Conv2DTranspose(self.input_shape[-1], 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(z, decoder_outputs, name="decoder")
        decoder.summary()
        return decoder

    def _build_model(self):
        encoder = self._build_encoder()
        decoder = self._build_decoder()
        vae = VarAutoEncoder(encoder, decoder)
        vae.compile(optimizer=Adam())
        
        return encoder, decoder, vae

    def train(self, generator: StackedMNISTData, epochs: np.int64 = 10):
        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """
        self.done_training = self.load_weights()

        if self.force_relearn or self.done_training is False:
            # Get hold of data, we only need the images, not the labels
            x_train, _ = generator.get_full_data_set(training=True)
            x_test, _ = generator.get_full_data_set(training=False)

            if self.mode == 'mono':
                # Process data for mono mode using only the first channel "red"
                x_train = x_train[:, :, :, [0]]
            
            elif self.mode == 'color':
                # Process data for color mode using all three channels
                pass

            # Fit model
            self.vae.fit(x=x_train, batch_size=1024, epochs=epochs)

            # Save weights and leave
            self.vae.save_weights(filepath=self.file_name)
            self.done_training = True

        return self.done_training

    def load_weights(self):
        # noinspection PyBroadException
        try:
            self.vae.load_weights(filepath=self.file_name)
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
        if self.mode == 'mono':
            x_test = x_test[:, :, :, [0]]
        elif self.mode == 'color':
            pass
        decoded_imgs = self.vae.predict(x_test)


        plt.figure(figsize=(16, 4))
        for i in range(num_images):
            # Original
            ax = plt.subplot(2, num_images, i + 1)
            if self.mode == 'mono':
                plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
            elif self.mode == 'color':
                plt.imshow(x_test[i].reshape(28, 28, 3)*255)
            ax.set_title('Original')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Reconstruction
            ax = plt.subplot(2, num_images, i + 1 + num_images)
            if self.mode == 'mono':
                plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
            elif self.mode == 'color':
                plt.imshow(decoded_imgs[i].reshape(28, 28, 3))
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
            if self.mode == 'mono':
                plt.imshow(generated_images[i].numpy().reshape(28, 28), cmap='gray')
            elif self.mode == 'color':
                plt.imshow(generated_images[i].numpy().reshape(28, 28, 3))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    
    def calculate_probalilities(self, images):
        """
        Calculate probability of seeing the image. This is done by calculating the binary cross-entropy loss between
        10 000 generated random images and the image to estimate the probability of seeing.
        """

        # Generate 10 000 random images
        z = np.random.randn(10000, self.encoding_dim)
        generated_images = self.decoder(z)

        # Calculate binary cross-entropy loss
        loss = keras.losses.binary_crossentropy(images, generated_images)

        # Mean over all pixels and inverse logarithm for probabilities
        probabilities = np.exp(np.mean(loss, axis=(1, 2))) 

        return probabilities
       


    def display_anomalies(self, generator: StackedMNISTData, num_images=10):
        """
        Display images with the highest reconstruction error
        """
        x_test, y_test = generator.get_full_data_set(training=False)
        if self.mode == 'mono':
            x_test = x_test[:, :, :, [0]]
        elif self.mode == 'color':
            pass
        
        # Find the images with the lowest probability to occur
        probabilities = self.calculate_probalilities(x_test)
        idxs = np.argsort(probabilities)[:num_images]
        anomalous_images = x_test[idxs]
        anomalous_labels = y_test[idxs]
        
        plt.figure(figsize=(16, 4))
        plt.suptitle('Images with the lowest reconstruction probability')
        for i, idx in enumerate(idxs):
            ax = plt.subplot(1, num_images, i + 1)
            if self.mode == 'mono':
                plt.imshow(anomalous_images[i].reshape(28, 28), cmap='gray')
                label = str(anomalous_labels[i])[-1]
                ax.set_title(f"{label}")
            elif self.mode == 'color':
                plt.imshow(anomalous_images[i].reshape(28, 28, 3)*255)
                ax.set_title(f"{anomalous_labels[i]}")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

def print_missing_digit(gen: StackedMNISTData):
    all_labels = [str(i) for i in range(10)]
    for label in gen.train_labels:
        for digit in str(label):
            if digit in all_labels:
                all_labels.remove(digit)
    print(f"Missing digits: {all_labels}")

if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=2048)
    print_missing_digit(gen)
    
    # Train the model
    net = VarAutoEncoderNet(mode='color', encoding_dim=20, file_name="./models/vae/vae_missing_color")
    net.train(generator=gen, epochs=10)

    # Reconstruct some images
    net.show_reconstructions(generator=gen, num_images=10)

    # Generate new images
    net.show_generated_images(num_images=10)

    # Display images with the highest reconstruction error
    net.display_anomalies(generator=gen, num_images=10)


"""
================================================================
Conclusion about the performance of the variational auto encoder
================================================================

--------------
Reconstruction
--------------
The reconstructions are quite good. The images are recognizable, but they are not as sharp as the in the auto-encoder images.
This could be because of the randomness injected into the latent space.

----------
Generation
----------
A lot better than the ordinary auto encoder. The images generated looks like actual digits that someone could have written.

-----------------
Anomaly Detection
-----------------
I'm not sure I'm doing it correctly. It's supposed to find the missing value '8', but it leans more towards '1' and '7'.


--------
Comments
--------

Key differences between the variational auto-encoder and the auto-encoder:

- The variational auto-encoder has a stochastic element in the latent space. This means that the latent space is not a single point, but a distribution. 
  This is achieved by having the encoder output the mean and standard deviation of the distribution, and then sampling from this distribution to get a point in the latent space.

- The variational auto-encoder has a regularization term in the loss function that encourages the distribution in the latent space to be close to a standard normal distribution.
  This is realized by adding the KL divergence between the distribution in the latent space and a standard normal distribution to the loss function.


------------------
Pre-trained models
------------------
Pre-trained models include:
'./models/vae/vae'                  |    one-channel images    |   complete    |    latent dimension: 24
'./models/vae/vae_big'              |    one-channel images    |   complete    |    latent dimension: 128
'./models/vae/vae_small'            |    one-channel images    |   complete    |    latent dimension: 5
'./models/vae/vae_missing'          |    one-channel images    |   missing     |    latent dimension: 12
'./models/vae/vae_color'            |    three-channel images  |   complete    |    latent dimension: 15
'./models/vae/vae_missing_color'    |    three-channel images  |   missing     |    latent dimension: 20

"""