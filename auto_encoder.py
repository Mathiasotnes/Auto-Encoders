from stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape, Input
from tensorflow.keras.optimizers.legacy import Adam
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
    
    def encode(self, images):
        return self.encoder.predict(images)

    def decode(self, encoded_imgs):
        return self.decoder.predict(encoded_imgs)

    def generate_new_images(self, num_images=10):
        # Generate new images (assuming a Gaussian distribution for the encoding)
        encoded_imgs = np.random.normal(size=(num_images, self.encoding_dim))
        generated_images = self.decode(encoded_imgs)
        return generated_images



if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    net = AutoEncoderNet(encoding_dim=128)
    net.train(generator=gen, epochs=2)

    # I have no data generator (VAE or whatever) here, so just use a sampled set
    # img, labels = gen.get_random_batch(training=True,  batch_size=25000)
    # cov = net.check_class_coverage(data=img, tolerance=.98)
    # pred, acc = net.check_predictability(data=img, correct_labels=labels)
    # print(f"Coverage: {100*cov:.2f}%")
    # print(f"Predictability: {100*pred:.2f}%")
    # print(f"Accuracy: {100 * acc:.2f}%")
    print('Script completed.')