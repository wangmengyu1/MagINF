

from keras.models import Model
from keras.layers import Input, core, Dropout, concatenate
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers import LSTM,Dense,Flatten


def Unet(nClasses,optimizer =None,input_length=1440, nChannels=3):  

    inputs = Input((input_length, nChannels))  # #  filters   kernel_size
    print("input",inputs.shape)
    conv1 = Conv1D(16, 64, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    print("conv1", conv1.shape)
    conv1 = Conv1D(16, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    print("conv1",conv1.shape)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    print("pool1",pool1.shape)

    conv2 = Conv1D(32, 64, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Dropout(0.1)(conv2)  
    conv2 = Conv1D(32, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Dropout(0.1)(conv3)  
    conv3 = Conv1D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(128, 64, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    print("conv4", conv4.shape)
    conv4 = Conv1D(128, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    print("conv4",conv4.shape)
    pool4 = MaxPooling1D(pool_size=2)(conv4)
    print("pool4",pool4.shape)

    conv5 = Conv1D(256, 64, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    print("conv5", conv5.shape)
    conv5 = Dropout(0.4)(conv5)  
    print("conv5", conv5.shape)
    conv5 = Conv1D(256, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    print("conv5", conv5.shape)
    up1 = Conv1D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv5))
    print("up1",up1.shape)
    merge1 = concatenate([up1, conv4], axis=-1)
    print("merge1",merge1.shape)
    conv6 = Conv1D(128, 64, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    print("conv6", conv6.shape)
    conv6 = Conv1D(128, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    print("conv6", conv6.shape)

    up2 = Conv1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv6))
    print("up2",up2.shape)
    merge2 = concatenate([up2, conv3], axis=-1)
    print("merge2",merge2.shape)
    conv7 = Conv1D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
    print("conv7", conv7.shape)
    conv7 = Dropout(0.1)(conv7)  
    print("conv7", conv7.shape)
    conv7 = Conv1D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    print("conv7",conv7.shape)

    up3 = Conv1D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv7))
    print("up3",up3.shape)
    merge3 = concatenate([up3, conv2], axis=-1)
    print("merge3",merge3.shape)
    conv8 = Conv1D(32, 64, activation='relu', padding='same', kernel_initializer='he_normal')(merge3)
    print("conv8", conv8.shape)
    conv8 = Dropout(0.1)(conv8)  
    print("conv8", conv8.shape)
    conv8 = Conv1D(32, 64, activation='relu', padding='same')(conv8)
    print("conv8", conv8.shape)

    up4 = Conv1D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv8))
    print("up4",up4.shape)
    merge4 = concatenate([up4, conv1], axis=-1)
    print("merge4", merge4.shape)
    conv9 = Conv1D(16, 64, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)
    print("conv9", conv9.shape)
    conv9 = Conv1D(16, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    print("conv9", conv9.shape)

    conv10 = Conv1D(nClasses, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    print("conv10", conv10.shape)
    conv10 = core.Reshape((nClasses, input_length))(conv10)
    print("conv10", conv10.shape)
    conv10 = core.Permute((2, 1))(conv10)
    print("conv10",conv10.shape)

    conv11 = core.Activation('softmax')(conv10)

    model = Model(inputs=inputs, outputs=conv11)
    if not optimizer is None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model

if __name__ == '__main__':
    print("-------------2----------------")
    print('\nSummarize the model:\n')
    model = Unet(4)  # 3 ？  ===>2 ??
    model.summary()
    print('\nEnd for summary.\n')

