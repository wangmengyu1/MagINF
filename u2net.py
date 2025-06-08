import tensorflow as tf
from tensorflow.keras import layers, Model, Input


class REBNCONV(tf.keras.layers.Layer):
    def __init__(self, out_ch, dilation=1, **kwargs):
        super(REBNCONV, self).__init__(**kwargs)
        self.out_ch = out_ch
        self.dilation = dilation
        self.conv = layers.Conv1D(out_ch, kernel_size=3, padding='same', dilation_rate=dilation)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

    def get_config(self):
        config = super(REBNCONV, self).get_config()
        config.update({
            "out_ch": self.out_ch,
            "dilation": self.dilation,
        })
        return config


def upsample_like(source, target):
    target_len = tf.shape(target)[1]
    return tf.image.resize(tf.expand_dims(source, axis=2), [target_len, 1], method='bilinear')[:, :, 0, :]


def RSU5_1D_model(input_shape=(1440, 3), mid_ch=12, out_ch=2):
    inputs = Input(shape=input_shape)

    rebnconvin = REBNCONV(out_ch)
    hxin = rebnconvin(inputs)

    rebnconv1 = REBNCONV(mid_ch)
    hx1 = rebnconv1(hxin)
    hx = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(hx1)

    rebnconv2 = REBNCONV(mid_ch)
    hx2 = rebnconv2(hx)
    hx = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(hx2)

    rebnconv3 = REBNCONV(mid_ch)
    hx3 = rebnconv3(hx)
    hx = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(hx3)

    rebnconv4 = REBNCONV(mid_ch)
    hx4 = rebnconv4(hx)

    rebnconv5 = REBNCONV(mid_ch, dilation=2)
    hx5 = rebnconv5(hx4)

    rebnconv4d = REBNCONV(mid_ch)
    hx4d = rebnconv4d(tf.concat([hx5, hx4], axis=-1))
    hx4dup = upsample_like(hx4d, hx3)

    rebnconv3d = REBNCONV(mid_ch)
    hx3d = rebnconv3d(tf.concat([hx4dup, hx3], axis=-1))
    hx3dup = upsample_like(hx3d, hx2)

    rebnconv2d = REBNCONV(mid_ch)
    hx2d = rebnconv2d(tf.concat([hx3dup, hx2], axis=-1))
    hx2dup = upsample_like(hx2d, hx1)

    rebnconv1d = REBNCONV(out_ch)
    hx1d = rebnconv1d(tf.concat([hx2dup, hx1], axis=-1))

    out = layers.Add()([hx1d, hxin])

    model = Model(inputs=inputs, outputs=out, name="RSU5_1D")
    return model
if __name__ == "__main__":
    model = RSU5_1D_model(input_shape=(1440, 3), mid_ch=12, out_ch=2)
    model.summary()
