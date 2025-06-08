import tensorflow as tf
from tensorflow.keras import layers, models

def DenseASPP1D(n_classes=2, input_length=1440, n_channels=3):
    """
    DenseASPP for 1D time series data.
    :param n_classes: number of output classes
    :param input_length: length of the time series (e.g., 1440)
    :param n_channels: number of input channels (e.g., 3)
    :return: tf.keras.Model
    """
    def dilated_conv_block(x, filters, kernel_size=3, rate=1):
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=rate,
                          kernel_initializer='he_normal')(x)
        return x

    inputs = tf.keras.Input(shape=(input_length, n_channels))

    # Encoder: simple conv block
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    c5 = layers.Conv1D(256, 3, padding='same', activation='relu')(x)  # Output of encoder

    # DenseASPP blocks
    d3 = dilated_conv_block(c5, 256, 1)
    d3 = dilated_conv_block(d3, 64, 3, rate=3)

    d4 = tf.keras.layers.Concatenate()([c5, d3])
    d4 = dilated_conv_block(d4, 256, 1)
    d4 = dilated_conv_block(d4, 64, 3, rate=6)

    d5 = tf.keras.layers.Concatenate()([c5, d3, d4])
    d5 = dilated_conv_block(d5, 256, 1)
    d5 = dilated_conv_block(d5, 64, 3, rate=12)

    d6 = tf.keras.layers.Concatenate()([c5, d3, d4, d5])
    d6 = dilated_conv_block(d6, 256, 1)
    d6 = dilated_conv_block(d6, 64, 3, rate=18)

    d7 = tf.keras.layers.Concatenate()([c5, d3, d4, d5, d6])
    d7 = dilated_conv_block(d7, 256, 1)
    d7 = dilated_conv_block(d7, 64, 3, rate=24)

    x = tf.keras.layers.Concatenate()([c5, d3, d4, d5, d6, d7])
    x = layers.Conv1D(n_classes, 1, padding='same')(x)

    # Optional: upsample back to original length (if pooling reduced it)
    # Since we used two MaxPool1D (each /2), total factor is 4 -> upsample by 4
    x = layers.UpSampling1D(size=4)(x)

    outputs = layers.Softmax()(x)

    return tf.keras.Model(inputs, outputs, name='DenseASPP1D')
if __name__ == "__main__":
    model = DenseASPP1D(n_classes=2, input_length=1440, n_channels=3)
    model.summary()

    # import numpy as np
    # dummy_input = np.random.rand(1, 1440, 3).astype(np.float32)
    # dummy_output = model(dummy_input)
    # print("Output shape:", dummy_output.shape)
