# -*- coding: utf-8 -*-
# @时间 : 2024/3/25 10:17
# @作者 : wangmengyu
# @Email : 1179763088@qq.com
# @File : segnet.py
# @Project : unet_Resnet3
from types import MethodType

from keras import Input, Model
from keras.layers import ZeroPadding1D, Conv1D, BatchNormalization, UpSampling1D, Activation, SeparableConv1D, Reshape
import keras.backend as K



def _conv_block(inputs, filters, alpha, kernel= 5, strides= 1):


    filters = int(filters * alpha)
    # x = ZeroPadding1D(padding=1, name='conv1_pad'
    #                   )(inputs)
    # print("x",x.shape)
    x = Conv1D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    print("x", x.shape)
    x = BatchNormalization(axis=-1, name='conv1_bn')(x)
    return Activation('relu', name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=1, block_id=1):


    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    # x = ZeroPadding1D(1,
    #                   name='conv_pad_%d' % block_id)(inputs)
    # print("x1", x.shape)
    x = SeparableConv1D(kernel_size=5,
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id,filters=pointwise_conv_filters)(inputs)
    print("x1", x.shape)
    x = BatchNormalization(
        axis=-1, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation('relu', name='conv_dw_%d_relu' % block_id)(x)

    x = Conv1D(pointwise_conv_filters, 5,
               padding='same',
               use_bias=False,
               strides=1,
               name='conv_pw_%d' % block_id)(x)
    print("x1", x.shape)
    x = BatchNormalization(axis=-1,
                           name='conv_pw_%d_bn' % block_id)(x)
    return Activation('relu', name='conv_pw_%d_relu' % block_id)(x)


def get_mobilenet_encoder(input_height=1440,
                           channels=3):

    # todo add more alpha and stuff



    alpha = 1.0
    depth_multiplier = 1
    dropout = 1e-3

    img_input = Input(shape=(input_height, channels))

    x = _conv_block(img_input, 32, alpha, strides=2)
    print("x3",x.shape)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    print("x3", x.shape)
    f1 = x

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=2, block_id=2)
    print("x3", x.shape)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    print("x3", x.shape)
    f2 = x

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides= 2, block_id=4)
    print("x3", x.shape)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    print("x3", x.shape)
    f3 = x

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=2, block_id=6)
    print("x3", x.shape)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    print("x3", x.shape)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    print("x3", x.shape)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    print("x3", x.shape)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    print("x3", x.shape)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    print("x3", x.shape)
    f4 = x

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=2, block_id=12)
    print("x3", x.shape)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    print("x3", x.shape)
    f5 = x



    return img_input, [f1, f2, f3, f4, f5]






def segnet_decoder(f, n_classes, n_up=4):

    assert n_up >= 2

    o = f
    print("o",o.shape)
    # o = (ZeroPadding1D(1))(o)
    # print("o", o.shape)
    o = (Conv1D(512,5, padding='same'))(o)
    print("o", o.shape)
    o = (BatchNormalization())(o)

    o = (UpSampling1D(2))(o)
    print("o1",o.shape)
    # o = (ZeroPadding1D(1))(o)
    # print("o1", o.shape)
    o = (Conv1D(256, 5, padding='same'))(o)
    print("o1", o.shape)
    o = (BatchNormalization())(o)

    for _ in range(n_up-2):
        o = (UpSampling1D(2))(o)
        # o = (ZeroPadding1D(1))(o)
        o = (Conv1D(128,5, padding='same'))(o)
        o = (BatchNormalization())(o)

    o = (UpSampling1D(2))(o)
    print("o2", o.shape)
    # o = (ZeroPadding1D(1))(o)
    # print("o2", o.shape)
    o = (Conv1D(64, 5, padding='same', name="seg_feats"))(o)
    print("o2", o.shape)
    o = (BatchNormalization())(o)

    o = Conv1D(32, 5, padding='same'
               )(o)
    print("o2", o.shape)


    o=(UpSampling1D(2))(o)
    print("o3",o.shape)
    o=(Conv1D(16,3,padding='same'))(o)
    o=(BatchNormalization())(o)

    o=Conv1D(n_classes,3,padding='same')(o)

    return o
def get_segmentation_model(input, output):

    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape


    output_height = o_shape[1]

    input_height = i_shape[1]

    n_classes = 2
    o = (Reshape((output_height, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)

    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height

    model.model_name = ""

    # model.train = MethodType(train, model)
    # model.predict_segmentation = MethodType(predict, model)
    # model.predict_multiple = MethodType(predict_multiple, model)
    # model.evaluate_segmentation = MethodType(evaluate, model)

    return model
def _segnet(n_classes, encoder,  input_height=1440,
            encoder_level=4, channels=3):

    img_input, levels = encoder(
        input_height=input_height,channels=channels)

    feat = levels[encoder_level]
    o = segnet_decoder(feat, n_classes, n_up=4)
    model = get_segmentation_model(img_input, o)

    return model
def mobilenet_segnet(n_classes, input_height=1440,
                     encoder_level=4, channels=3):

    model = _segnet(n_classes, get_mobilenet_encoder,
                    input_height=input_height,
                    encoder_level=encoder_level, channels=channels)
    model.model_name = "mobilenet_segnet"
    return model


if __name__ == '__main__':
    # m = vgg_segnet(101)
    # m = segnet(101)
    m = mobilenet_segnet( 2 )
    m.summary()
    # from keras.utils import plot_model
    # plot_model( m , show_shapes=True , to_file='model.png')