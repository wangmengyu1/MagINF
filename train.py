

# -*- coding: utf-8 -*-
# @时间 : 2024/3/25 10:56
# @作者 : wangmengyu
# @Email : 1179763088@qq.com
# @File : segnet_train.py
# @Project : unet_Resnet3# -*- coding: utf-8 -*-
# """
# Created on Sun Apr 21 13:52:07 2019
#
# @author: Administrator
#
# train.py: train models
#
# """
#
# from Unet import Unet,CNN4,CNN4_LSTM
# from Unet import Unet

# import LoadBatches1D_1

from tensorflow import keras
# from tensorflow import keras
from tensorflow.keras import optimizers,activations
import warnings
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

# 59
tf.compat.v1.disable_v2_behavior()

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCR_GPU_ALLOW_GROWTH"] = "true"  # 
# tf.compat.v1.GPUOptions.per_preocess_gpu_memory_fraction = 0.7  

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
exp_id = 'FEU-net'  # 

weights = [0.5, 1.5,2,1.5]  



import tensorflow as tf

def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):

        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        weighted_losses = y_true * tf.math.log(y_pred) * weights
        loss = -tf.reduce_sum(weighted_losses, axis=-1)  

        return tf.reduce_mean(loss) 

    return loss

def lr_schedule(epoch):

    # lr = 0.0003565
    lr = 0.00005

    # lr = 0.004
    # decay_rate = 0.9  # 衰减率
    # decay_step = 10  # 每10个epoch衰减一次
    # if epoch >= 20:
    #     lr = 0.0002
    # # 0.0003565
    # if  60<= epoch < 100:
    #     lr = 0.00005
    # # if 60<=epoch<100:
    # #     lr=0.00001
    # if epoch >= 100:
    #     lr = 0.00001
    print('Learning rate: ', lr)
    return lr


start = time.time()


train_sigs_path = '../data/train_data/'
train_segs_path = '../data/train_label/'
val_sigs_path = '../data/val_data/'
val_segs_path = '../data/val_label/'

SAVE_DIR = '../image/image_样本论文' + '/{}'.format(exp_id)
if not os.path.exists(os.path.join(SAVE_DIR)):
    os.mkdir(os.path.join(SAVE_DIR))
print("SAVE_DIR", SAVE_DIR)
train_batch_size = 16  
n_classes = 4  
input_length = 1440 
optimizer_name = optimizers.Adam(lr_schedule(0),clipnorm=1)  
PATIENCE = 50  
val_batch_size = 64  # ?
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
input_shape=(1440,3)

num_classes = 4

from depthwise2 import depthwise
'''
pspnet
'''
# from PSPnet import PSPNet
# model = PSPNet(nClasses=4, input_length=1440, n_channels=3)

'''
unet
'''
# from Unet import Unet
# model = Unet(4)
'''
segnet
'''
# from segnet import mobilenet_segnet
# model=mobilenet_segnet( 4 )

'''
deeplab
'''
# from deeplab import DeepLabV3
# model = DeepLabV3(input_shape=(1440, 3), depth=8, dropout=0)

'''
denseaspp
'''
# from denseaspp import DenseASPP1D
# model = DenseASPP1D(n_classes=4, input_length=1440, n_channels=3)
'''
swin_unet
'''

# from swin_unet import NestNet1D
# model = NestNet1D(input_shape=(1440, 3), n_labels=4, using_deep_supervision=False)
'''
resnet
'''
# from resunet import ResUNet1D
# sequence_length = 1440
# num_features = 3
# num_classes = 4
# model = ResUNet1D(filters=32, sequence_length=sequence_length, num_features=num_features, num_classes=num_classes)

'''
u^2net
'''
# from u2net import RSU5_1D_model
#
# model = RSU5_1D_model(input_shape=(1440, 3), mid_ch=12, out_ch=4)
model.compile(loss=weighted_categorical_crossentropy(weights),
              optimizer=optimizer_name,
              metrics=['accuracy'])#,loss=focal_loss(gamma=2., alpha=alpha_values
# metrics = ['acc'])
#loss=focal_loss(gamma=2., alpha=alpha_values)
model.summary()  

print("e-code={}".format(exp_id))
print("PATIENCE={}acc".format(PATIENCE))

output_length = 1440  
import LoadBatches1D_1
import Batch_4

G = LoadBatches1D_1.SigSegmentationGenerator(train_sigs_path, train_segs_path, train_batch_size, n_classes, output_length)

G2 = LoadBatches1D_1.SigSegmentationGenerator(val_sigs_path, val_segs_path, val_batch_size, n_classes, output_length)
checkpointer = [keras.callbacks.ModelCheckpoint(filepath='{}/bmodel.h5'.format(SAVE_DIR), monitor='val_acc', mode='max',
                                                save_best_only=True),
                keras.callbacks.EarlyStopping(monitor='val_acc', patience=PATIENCE)]



history = model.fit_generator(G, 5972 // train_batch_size, validation_data=G2,
                              validation_steps=int(746 / val_batch_size), epochs=150,
                              callbacks=[checkpointer, lr_scheduler])  
                              # callbacks=checkpointer)  

plt.figure()


plt.plot(history.history['acc']) 

plt.plot(history.history.get('val_accuracy') or history.history.get('val_acc'))

plt.title('Model accuracy {}'.format(exp_id))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.grid(True)


plt.savefig('{}/Accuracy.png'.format(SAVE_DIR)) 


txt_name2 = "acc_{}.txt".format(exp_id)
this_file = open(SAVE_DIR + "/" + txt_name2, "w")
this_file.write("acc")
this_file.write(str(history.history['acc']))
this_file.write("\n")
this_file.write("val_acc")
this_file.write(str(history.history['val_acc']))
this_file.write("\n")
this_file.close()
print("END END END")


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss {}'.format(exp_id))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.grid(True)

# plt.savefig('_fig/Loss.png')
# plt.savefig('./results/fig/Loss.png')  
plt.savefig('{}/Loss.png'.format(SAVE_DIR))




txt_name = "loss_{}.txt".format(exp_id)
this_file = open(SAVE_DIR + "/" + txt_name, "w")
this_file.write("loss")
this_file.write(str(history.history['loss']))
this_file.write("\n")
this_file.write("val_loss")
this_file.write(str(history.history['val_loss']))
this_file.write("\n")
this_file.close()




elapsed_time = time.time() - start
print(elapsed_time)
print("This process took approximately {} minutes".format(elapsed_time / 60))


print("--------Start evaluating the test set--------")

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, ConfusionMatrixDisplay, classification_report
import LoadBatches1D_1


test_geo_path = '../data/test_data/'
test_label_path = '../data/test_label/'

test_geo_files = os.listdir(test_geo_path)
test_label_files = os.listdir(test_label_path)

custom_loss = weighted_categorical_crossentropy(weights)

num_classes = 4


SAVE_DIR = '../image/image' + '/{}'.format(exp_id)
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
print("SAVE_DIR", SAVE_DIR)
from tensorflow.keras.models import load_model

from wtconv1 import WTConv1D
from FFT import FourierUnit
from mamba1 import Mamba,MambaBlock
from CNN_Transformer import SwinTransformerBlock
from u2net import REBNCONV
best_model_path = '{}/bmodel.h5'.format(SAVE_DIR)
model = load_model(best_model_path, compile=False,custom_objects={
    'weighted_categorical_crossentropy': custom_loss,'WTConv1D':WTConv1D,'FourierUnit':FourierUnit,'SwinTransformerBlock':SwinTransformerBlock,'Mamba':Mamba,'REBNCONV':REBNCONV
})
SAVE_DIR = '../image/image' + '/{}'.format(exp_id)


test_geo_files = os.listdir(test_geo_path)
test_label_files = os.listdir(test_label_path)

label_test = []  # truth label
label_pred = []  # Predictive label
probabilities = [[] for _ in range(4)]  # Store the predicted probabilities of the 4 categories


if len(test_geo_files) == len(test_label_files):
    for i in range(len(test_geo_files)):
        select = test_geo_files[i]
        if select == test_label_files[i]:
            # retrieve data
            a_geo = np.load(test_geo_path + select)  # shape: (1440, 3)
            a_label = np.load(test_label_path + select).flatten()  # Flatten to a one-dimensional array
            label_test.extend(a_label.tolist())  # Storing Real Labels

            # Construct model input: add batch dimension -> shape: (1, 1440, 3)
            X_test = np.expand_dims(a_geo, axis=0)

            # carry out forecasting
            a_pred = model.predict(X_test)

            # Get prediction categories (for each time step)
            a_pred_array = np.argmax(a_pred[0], axis=-1)
            label_pred.extend(a_pred_array.tolist())

            # Record the predicted probability for each category at each time step
            for j in range(4):
                probabilities[j].extend(a_pred[0][:, j].tolist())

            # Progress log
            if i % 100 == 0:
                print(f"{i + 1}/{len(test_geo_files)}")
# ========================
# 3. Calculation of assessment indicators
# ========================
print("Final Length of true labels:", len(label_test))
print("Final Length of predicted labels:", len(label_pred))

if len(label_test) != len(label_pred):
    raise ValueError(f"Length mismatch: label_test({len(label_test)}) != label_pred({len(label_pred)})")

# Calculation accuracy
acc = accuracy_score(label_test, label_pred)
print("Overall Accuracy:", acc)

# Calculate Precision, Recall, F1-score (for 4 classifications)
precision_per_class = precision_score(label_test, label_pred, average=None, zero_division=0)
recall_per_class = recall_score(label_test, label_pred, average=None, zero_division=0)
f1_per_class = f1_score(label_test, label_pred, average=None, zero_division=0)

# Calculate Precision, Recall, F1-score
precision_macro = precision_score(label_test, label_pred, average='weighted', zero_division=0)
recall_macro = recall_score(label_test, label_pred, average='weighted', zero_division=0)
f1_macro = f1_score(label_test, label_pred, average='weighted', zero_division=0)

# Print indicators by category
print("\nPer-Class Metrics:")
for cls in range(4):
    print(f"Class {cls}: Precision: {precision_per_class[cls]:.4f}, Recall: {recall_per_class[cls]:.4f}, F1 Score: {f1_per_class[cls]:.4f}")

print("\nOverall Metrics:")
print(f"Overall Precision: {precision_macro:.4f}")
print(f"Overall Recall: {recall_macro:.4f}")
print(f"Overall F1 Score: {f1_macro:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(label_test, label_pred, zero_division=0))

# ========================
# 4. Calculate IoU (Intersection over Union)
# ========================
num_classes = 4  
conf_matrix = confusion_matrix(label_test, label_pred, labels=np.arange(num_classes))

iou_per_class = []
for i in range(num_classes):
    tp = conf_matrix[i, i]
    fp = conf_matrix[:, i].sum() - tp
    fn = conf_matrix[i, :].sum() - tp
    iou = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0
    iou_per_class.append(iou)

# Calculate mIoU
miou = np.mean(iou_per_class)

print("\nPer-Class IoU:")
for i, iou in enumerate(iou_per_class):
    print(f"Class {i}: IoU: {iou:.4f}")
print(f"mIoU: {miou:.4f}")