import os
import matplotlib as mpl
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn import preprocessing as prep
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["TF_FORCR_GPU_ALLOW_GROWTH"] = "true"  # Try allocating GPU memory

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Set font for non-ASCII characters (if needed)
font = FontProperties(fname='/usr/share/fonts/chinese/Fonts/simhei.ttf', size=16)

experiment_folder = './compare_jijian/'
model_files = [f for f in os.listdir(experiment_folder) if f.endswith('.h5')]

test_geo_path = '../data/test_data/'
test_label_path = '../data/test_label/'

geo_files = os.listdir(test_geo_path)
label_files = os.listdir(test_label_path)

from CNN_Transformer import SwinTransformerBlock, AdaptiveFusionBlock, CATM
from FFT import FourierUnit
from mamba1 import Mamba, MambaBlock
from u2net import REBNCONV

# 4-class weights
weights = [0.5, 1.5, 2, 1.5]  # Adjust weights based on actual data

def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        weighted_losses = y_true * tf.math.log(y_pred) * weights
        loss = -tf.reduce_sum(weighted_losses, axis=-1)
        return tf.reduce_mean(loss)

    return loss

custom_loss = weighted_categorical_crossentropy(weights)

models = [
    load_model('./compare_sample1/DeepLab.h5', compile=False,
               custom_objects={'weighted_categorical_crossentropy': custom_loss, 'CATM': CATM,
                               'SwinTransformerBlock': SwinTransformerBlock, 'AdaptiveFusionBlock': AdaptiveFusionBlock,
                               'Mamba': Mamba, 'REBNCONV': REBNCONV, 'FourierUnit': FourierUnit}),
    load_model('./compare_sample1/DenseAspp.h5', compile=False,
               custom_objects={'weighted_categorical_crossentropy': custom_loss, 'CATM': CATM,
                               'SwinTransformerBlock': SwinTransformerBlock, 'AdaptiveFusionBlock': AdaptiveFusionBlock,
                               'Mamba': Mamba, 'REBNCONV': REBNCONV, 'FourierUnit': FourierUnit}),
    load_model('./compare_sample1/PSPnet.h5', compile=False,
               custom_objects={'weighted_categorical_crossentropy': custom_loss, 'CATM': CATM,
                               'SwinTransformerBlock': SwinTransformerBlock, 'AdaptiveFusionBlock': AdaptiveFusionBlock,
                               'Mamba': Mamba, 'REBNCONV': REBNCONV, 'FourierUnit': FourierUnit}),
    load_model('./compare_sample1/ResU-net.h5', compile=False,
               custom_objects={'InstanceNormalization': tfa.layers.InstanceNormalization}),
    load_model('./compare_sample1/Segnet.h5', compile=False,
               custom_objects={'weighted_categorical_crossentropy': custom_loss, 'CATM': CATM,
                               'SwinTransformerBlock': SwinTransformerBlock, 'AdaptiveFusionBlock': AdaptiveFusionBlock,
                               'Mamba': Mamba, 'REBNCONV': REBNCONV, 'FourierUnit': FourierUnit}),
    load_model('./compare_sample1/Swin U-net.h5', compile=False,
               custom_objects={'weighted_categorical_crossentropy': custom_loss, 'CATM': CATM,
                               'SwinTransformerBlock': SwinTransformerBlock, 'AdaptiveFusionBlock': AdaptiveFusionBlock,
                               'Mamba': Mamba, 'REBNCONV': REBNCONV, 'FourierUnit': FourierUnit}),
    load_model('./compare_sample1/U-net.h5', compile=False,
               custom_objects={'weighted_categorical_crossentropy': custom_loss, 'CATM': CATM,
                               'SwinTransformerBlock': SwinTransformerBlock, 'AdaptiveFusionBlock': AdaptiveFusionBlock,
                               'Mamba': Mamba, 'REBNCONV': REBNCONV, 'FourierUnit': FourierUnit}),
    load_model('./compare_sample1/U^2net.h5', compile=False,
               custom_objects={'weighted_categorical_crossentropy': custom_loss, 'CATM': CATM,
                               'SwinTransformerBlock': SwinTransformerBlock, 'AdaptiveFusionBlock': AdaptiveFusionBlock,
                               'Mamba': Mamba, 'REBNCONV': REBNCONV, 'FourierUnit': FourierUnit}),
]

model_names = [
    'DeepLab',
    'DenseAspp',
    'PSPnet',
    'ResU-net',
    'Segnet',
    'Swin U-net',
    'U-net',
    'U^2net',
]

def draw_interference_with_red_line(ax, data_line, labels, label=None):
    """Draw different types of interference events with background fill and red line, color and legend by label"""
    in_event = False
    start_index = 0
    current_class = None
    first_drawn = {1: True, 2: True, 3: True}  # Ensure each legend is drawn only once

    class_info = {
        1: {'color': '#d15254', 'label': 'HVDC'},        # Red
        2: {'color': '#fab733', 'label': 'infrastructure'},  # Orange
        3: {'color': '#9336fd', 'label': 'vehicle'},      # Purple
    }

    for i in range(len(labels)):
        lbl = labels[i]
        if lbl != 0:
            if not in_event:
                start_index = i
                in_event = True
                current_class = lbl
            elif lbl != current_class:
                info = class_info.get(current_class, {})
                ax.axvspan(start_index, i, color=info.get('color', 'gray'), alpha=0.3)
                ax.plot(range(start_index, i), data_line[start_index:i], color='red', linewidth=1,
                        label=info.get('label') if first_drawn.get(current_class, False) else None)
                first_drawn[current_class] = False
                start_index = i
                current_class = lbl
        else:
            if in_event:
                info = class_info.get(current_class, {})
                ax.axvspan(start_index, i, color=info.get('color', 'gray'), alpha=0.3)
                ax.plot(range(start_index, i), data_line[start_index:i], color='red', linewidth=1,
                        label=info.get('label') if first_drawn.get(current_class, False) else None)
                first_drawn[current_class] = False
                in_event = False
                current_class = None

    if in_event:
        info = class_info.get(current_class, {})
        ax.axvspan(start_index, len(data_line), color=info.get('color', 'gray'), alpha=0.3)
        ax.plot(range(start_index, len(data_line)), data_line[start_index:], color='red', linewidth=1,
                label=info.get('label') if first_drawn.get(current_class, False) else None)

def plot_single_model_prediction(geo_data, pred_probs, a_label, select, i, total, model_name):
    """Plot 3 components + model prediction, distinguish two classes with filled color"""
    component_names = ['Z', 'H', 'D']
    colors = ['black', 'black', 'black']

    fig, axs = plt.subplots(4, 1, figsize=(16, 10), dpi=300, sharex=True)
    time_steps = range(len(geo_data))

    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    global_min = np.min(geo_data)
    global_max = np.max(geo_data)

    for comp_idx in range(3):
        ax = axs[comp_idx]
        ax.plot(geo_data[:, comp_idx], color=colors[comp_idx], linewidth=1, label='BACKGROUND')
        draw_interference_with_red_line(ax, geo_data[:, comp_idx], a_label, label='INTERFERENCE')
        ax.set_ylabel(component_names[comp_idx], size=16, font=font)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim([0, len(geo_data)])
        ax.set_ylim([global_min, global_max])

    axs[0].legend(loc='upper right')



    ax = axs[3]
   


    class_colors = ['#0ddbf5', '#d15254', '#fab733', '#9336fd'] 
    class_labels = ['Probability of BACKGROUND', 'Probability of HVDC', 'Probability of INFRASTRUCTURE', 'Probability of VEHICLE']
    for class_idx in range(4):
        class_prob = np.clip(pred_probs[:, class_idx], 0, 1)
        ax.fill_between(time_steps, 0, class_prob,
                        color=class_colors[class_idx],
                        alpha=0.6,
                        label=class_labels[class_idx])

    ax.legend(loc='upper right')  
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel(model_name, fontsize=16)
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='--', alpha=0.5)

    axs[-1].set_xlabel('Time (Minutes)', fontsize=12)



    parts = select.split('_')
    station_code = parts[1]
    instrument_code = parts[2]
    date = parts[4].split('.')[0]
    title_str = f"Station: {station_code}, Instrument: {instrument_code}, Date: {date[4:6]}/{date[6:]}/{date[:4]}"
   
    axs[0].set_title(title_str, fontsize=16, fontweight='bold')


    save_path = f"./results/new_plot_sample_model/{model_name}/{select.split('.')[0]}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[{model_name}] 图像已保存：{select} ({i+1}/{total})")




test_geo_files = os.listdir(test_geo_path)
test_label_files = os.listdir(test_label_path)

target_idx = 0  # Just change this line to change the model.
model = models[target_idx]
model_name = model_names[target_idx]

for i in range(len(test_geo_files)):
    select = test_geo_files[i]
    if select != test_label_files[i]:
        print("File mismatch, skipping.")
        continue

    geo_path = os.path.join(test_geo_path, select)
    label_path = os.path.join(test_label_path, select)

    a_geo = np.load(geo_path)  # (1440, 3)
    a_label = np.load(label_path)

    if a_geo.shape != (1440, 3):
        print(f"Skipping {select}, shape mismatch: {a_geo.shape}")
        continue

    a_geo_input = np.expand_dims(a_geo, axis=0)  # (1, 1440, 3)
    probs = model.predict(a_geo_input, batch_size=4)[0]
    plot_single_model_prediction(
        geo_data=a_geo,
        pred_probs=probs,
        a_label=a_label,
        select=select,
        i=i,
        total=len(test_geo_files),
        model_name=model_name
    )

    if i % 100 == 0:
        print(f"Processed {i + 1}/{len(test_geo_files)})")


