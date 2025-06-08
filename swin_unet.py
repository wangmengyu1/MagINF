
from keras import backend as K
from keras.layers import concatenate, Conv1DTranspose, Activation, Conv1D
from keras.layers import BatchNormalization, AvgPool1D, Input
from keras.models import Model


def conv_batchnorm_relu_block(input_tensor, nb_filter, kernel_size=3):
    x = Conv1D(nb_filter, kernel_size, padding='same')(input_tensor)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x

def NestNet1D(input_shape, n_labels, using_deep_supervision=False):
    nb_filter = [32, 64, 128, 256, 512]
    K.set_image_data_format("channels_last")

    inputs = Input(shape=input_shape, name='input_sequence')

    conv1_1 = conv_batchnorm_relu_block(inputs, nb_filter=nb_filter[0])
    pool1 = AvgPool1D(2, strides=2, name='pool1')(conv1_1)

    conv2_1 = conv_batchnorm_relu_block(pool1, nb_filter=nb_filter[1])
    pool2 = AvgPool1D(2, strides=2, name='pool2')(conv2_1)

    up1_2 = Conv1DTranspose(nb_filter[0], 2, strides=2, padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], axis=-1)
    conv1_2 = conv_batchnorm_relu_block(conv1_2, nb_filter=nb_filter[0])

    conv3_1 = conv_batchnorm_relu_block(pool2, nb_filter=nb_filter[2])
    pool3 = AvgPool1D(2, strides=2, name='pool3')(conv3_1)

    up2_2 = Conv1DTranspose(nb_filter[1], 2, strides=2, padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], axis=-1)
    conv2_2 = conv_batchnorm_relu_block(conv2_2, nb_filter=nb_filter[1])

    up1_3 = Conv1DTranspose(nb_filter[0], 2, strides=2, padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], axis=-1)
    conv1_3 = conv_batchnorm_relu_block(conv1_3, nb_filter=nb_filter[0])

    conv4_1 = conv_batchnorm_relu_block(pool3, nb_filter=nb_filter[3])
    pool4 = AvgPool1D(2, strides=2, name='pool4')(conv4_1)

    up3_2 = Conv1DTranspose(nb_filter[2], 2, strides=2, padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], axis=-1)
    conv3_2 = conv_batchnorm_relu_block(conv3_2, nb_filter=nb_filter[2])

    up2_3 = Conv1DTranspose(nb_filter[1], 2, strides=2, padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], axis=-1)
    conv2_3 = conv_batchnorm_relu_block(conv2_3, nb_filter=nb_filter[1])

    up1_4 = Conv1DTranspose(nb_filter[0], 2, strides=2, padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], axis=-1)
    conv1_4 = conv_batchnorm_relu_block(conv1_4, nb_filter=nb_filter[0])

    conv5_1 = conv_batchnorm_relu_block(pool4, nb_filter=nb_filter[4])

    up4_2 = Conv1DTranspose(nb_filter[3], 2, strides=2, padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], axis=-1)
    conv4_2 = conv_batchnorm_relu_block(conv4_2, nb_filter=nb_filter[3])

    up3_3 = Conv1DTranspose(nb_filter[2], 2, strides=2, padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], axis=-1)
    conv3_3 = conv_batchnorm_relu_block(conv3_3, nb_filter=nb_filter[2])

    up2_4 = Conv1DTranspose(nb_filter[1], 2, strides=2, padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], axis=-1)
    conv2_4 = conv_batchnorm_relu_block(conv2_4, nb_filter=nb_filter[1])

    up1_5 = Conv1DTranspose(nb_filter[0], 2, strides=2, padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], axis=-1)
    conv1_5 = conv_batchnorm_relu_block(conv1_5, nb_filter=nb_filter[0])

    output_1 = Conv1D(n_labels, 1, activation='sigmoid', name='output_1', padding='same')(conv1_2)
    output_2 = Conv1D(n_labels, 1, activation='sigmoid', name='output_2', padding='same')(conv1_3)
    output_3 = Conv1D(n_labels, 1, activation='sigmoid', name='output_3', padding='same')(conv1_4)
    output_4 = Conv1D(n_labels, 1, activation='sigmoid', name='output_4', padding='same')(conv1_5)

    if using_deep_supervision:
        model = Model(inputs=inputs, outputs=[output_1, output_2, output_3, output_4])
    else:
        model = Model(inputs=inputs, outputs=output_4)

    return model


if __name__ == '__main__':

    # from models.nestnet1d import NestNet1D
    model = NestNet1D(input_shape=(1440, 3), n_labels=4, using_deep_supervision=False)
    model.summary()
