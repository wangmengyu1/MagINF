from keras import Input, Model
from keras.layers import Conv1D, BatchNormalization, Dropout, MaxPooling1D, concatenate, UpSampling1D, Activation
from tensorflow.keras import backend as K

def atrous_spatial_pyramid_pooling(input_layer, depth, dropout):
    conv11_layer = Conv1D(depth, 1, activation='relu', padding="same", kernel_initializer='he_normal')(input_layer)
    conv11_layer = BatchNormalization()(conv11_layer)
    # conv11_layer = Dropout(dropout)(conv11_layer)

    atrous_conv1 = Conv1D(depth, 5, dilation_rate=6, activation='relu', padding="same", kernel_initializer='he_normal')(input_layer)
    atrous_conv1 = BatchNormalization()(atrous_conv1)
    # atrous_conv1 = Dropout(dropout)(atrous_conv1)

    atrous_conv2 = Conv1D(depth, 5, dilation_rate=12, activation='relu', padding="same", kernel_initializer='he_normal')(input_layer)
    atrous_conv2 = BatchNormalization()(atrous_conv2)
    # atrous_conv2 = Dropout(dropout)(atrous_conv2)

    atrous_conv3 = Conv1D(depth, 5, dilation_rate=18, activation='relu', padding="same", kernel_initializer='he_normal')(input_layer)
    atrous_conv3 = BatchNormalization()(atrous_conv3)
    # atrous_conv3 = Dropout(dropout)(atrous_conv3)

    return conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3

def DeepLabV3(input_shape, depth, dropout=0.25):
    input_layer = Input(shape=input_shape)

    # Block 1
    output_layer = Conv1D(depth, 5, activation='relu', padding="same", kernel_initializer='he_normal')(input_layer)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Conv1D(depth, 5, activation='relu', padding="same", kernel_initializer='he_normal')(output_layer)
    output_layer = BatchNormalization()(output_layer)
    output_layer = MaxPooling1D(pool_size=2)(output_layer)
    # output_layer = Dropout(dropout)(output_layer)

    # Block 2
    output_layer = Conv1D(depth * 2, 5, activation='relu', padding="same", kernel_initializer='he_normal')(output_layer)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Conv1D(depth * 2, 5, activation='relu', padding="same", kernel_initializer='he_normal')(output_layer)
    output_layer = BatchNormalization()(output_layer)
    output_layer = MaxPooling1D(pool_size=2)(output_layer)
    # output_layer = Dropout(dropout)(output_layer)

    # Block 3
    output_layer = Conv1D(depth * 4, 5, activation='relu', padding="same", kernel_initializer='he_normal')(output_layer)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Conv1D(depth * 4, 5, activation='relu', padding="same", kernel_initializer='he_normal')(output_layer)
    output_layer = BatchNormalization()(output_layer)
    output_layer = MaxPooling1D(pool_size=2)(output_layer)
    # output_layer = Dropout(dropout)(output_layer)

    # Block 4
    output_layer = Conv1D(depth * 8, 5, dilation_rate=2, activation='relu', padding="same", kernel_initializer='he_normal')(output_layer)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Conv1D(depth * 8, 5, dilation_rate=2, activation='relu', padding="same", kernel_initializer='he_normal')(output_layer)
    output_layer = BatchNormalization()(output_layer)
    # output_layer = Dropout(dropout)(output_layer)

    # Atrous Spatial Pyramid Pooling
    conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3 = atrous_spatial_pyramid_pooling(output_layer, depth * 8, dropout)

    # Block 6
    maxpooled_input = MaxPooling1D(pool_size=8)(input_layer)
    concatenated = concatenate([conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3, maxpooled_input])
    output_layer = Conv1D(3, 1, activation="relu", padding="same", kernel_initializer='he_normal')(concatenated)

    # Block 7
    # TensorFlow 2.x does not support 'bilinear' upsampling in 1D. So we use simple UpSampling1D instead.
    output_layer = UpSampling1D(size=8)(output_layer)

    output_layer = Activation("softmax")(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

if __name__ == '__main__':
    # Create DeepLabV3 model
    model = DeepLabV3(input_shape=(86400, 3), depth=8, dropout=0)

    # Print model summary
    model.summary()
