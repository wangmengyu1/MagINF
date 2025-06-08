import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv1D

# Dilated Convolution Block for 1D
def dilated_conv_block(inputs, filters, kernel_size=3, rate=1):
    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, padding='same',
                      dilation_rate=rate, kernel_initializer='he_normal')(x)
    return x

# Bidirectional GRU Block
def bi_gru_block(inputs, units):
    x = layers.Bidirectional(layers.GRU(units, return_sequences=True))(inputs)
    return x

# DenseASPP-like structure
def denseaspp_1d(inputs, num_classes):
    # Simulating the encoder output for 1D inputs
    c5 = dilated_conv_block(inputs, 256, 3)

    # First block rate=3, replace with BiGRU
    # d3 = bi_gru_block(c5, 64)
    d3 = dilated_conv_block(c5, 64, 3, rate=3)
    d3=Conv1D(64,3,padding='same')(d3)

    # Second block rate=6
    d4 = layers.Concatenate()([c5, d3])
    d4 = dilated_conv_block(d4, 256, 1)
    # d4 = dilated_conv_block(d4, 64, 3, rate=6)
    d4=Conv1D(64,3,padding='same')(d4)

    # Third block rate=12, replace with BiGRU
    d5 = layers.Concatenate()([c5, d3, d4])
    # d5 = bi_gru_block(d5, 64)
    # d5 = dilated_conv_block(d5, 64, 3, rate=12)
    d5=Conv1D(64,3,padding='same')(d5)

    # Fourth block rate=18
    d6 = layers.Concatenate()([c5, d3, d4, d5])
    # d6 = dilated_conv_block(d6, 256, 1)
    # d6 = dilated_conv_block(d6, 64, 3, rate=18)
    d6=Conv1D(256,1,padding='same')(d6)
    d6=Conv1D(64,3,padding='same')(d6)

    # Fifth block rate=24, replace with BiGRU
    d7 = layers.Concatenate()([c5, d3, d4, d5, d6])
    # d7 = bi_gru_block(d7, 64)
    # d7 = dilated_conv_block(d7, 64, 3, rate=24)
    d7=Conv1D(64,3,padding='same')(d7)

    # Final layer
    x = layers.Concatenate()([c5, d3, d4, d5, d6, d7])
    x = layers.Conv1D(num_classes, 1, strides=1, kernel_initializer='he_normal')(x)

    return x

if __name__ == "__main__":
    # Define input shape for time series data (86400 time steps, 3 features)
    input_shape = (86400, 3)
    num_classes = 4  # Adjust according to your problem

    # Create input layer
    inputs = layers.Input(shape=input_shape)

    # Build the DenseASPP-like model using function definitions
    outputs = denseaspp_1d(inputs, num_classes)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs, name='DenseASPP1D')

    # Compile and summarize the model
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()
