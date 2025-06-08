# -*- coding: utf-8 -*-


import os
import itertools
import numpy as np
from sklearn import preprocessing as prep
import tensorflow as tf

'''
Three signal components
'''
# Modifications
def getSigArr(path, sigNorm='minmax'):
    sig = np.load(path)

    return sig  # Directly return sig, used for three-component input
    # return np.expand_dims(sig, axis=1)  # Use this for single-component input

    # return sig

def getSegmentationArr(path, nClasses=4, output_length=1440, class_value=[0,1,2,3]):
    seg_labels = np.zeros([output_length, nClasses])
    seg = np.load(path)

    # Check data type
    if not isinstance(seg, np.ndarray):
        raise ValueError(f"Expected np.ndarray, but got {type(seg)}. File path: {path}")

    # Check data shape
    if seg.ndim != 1:
        raise ValueError(f"Expected 1D array, but got {seg.ndim}D array. File path: {path}")

    # Ensure data type is integer
    if not np.issubdtype(seg.dtype, np.integer):
        if np.issubdtype(seg.dtype, np.floating):
            # print(f"Warning: seg is of float type, converting to int")
            seg = seg.astype(int)
        else:
            raise ValueError(f"Expected seg to be of integer type, but got {seg.dtype}")

    # Generate one-hot labels
    for i in range(nClasses):
        seg_labels[:, i] = (seg == class_value[i]).astype(float)

    return seg_labels


def SigSegmentationGenerator(sigs_path, segs_path, batch_size, n_classes, output_length=1440):  # 1440 -- 86400
    sigs = [s for s in os.listdir(sigs_path) if s.endswith('.npy')]
    segmentations = [s for s in os.listdir(segs_path) if s.endswith('.npy')]

    # Ensure sigs and segmentations are sorted by filename
    sigs.sort()
    segmentations.sort()

    paired_sigs = []
    paired_segs = []

    for sig in sigs:
        sig_name = os.path.splitext(sig)[0]
        for seg in segmentations:
            seg_name = os.path.splitext(seg)[0]
            if sig_name == seg_name:
                paired_sigs.append(sigs_path + sig)
                paired_segs.append(segs_path + seg)
                break

    assert len(paired_sigs) == len(paired_segs)
    zipped = itertools.cycle(zip(paired_sigs, paired_segs))
    # print("Debug zipped data:", zipped)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            sig, seg = next(zipped)
            X.append(getSigArr(sig))  # Append directly
            Y.append(getSegmentationArr(seg, n_classes, output_length))

        yield np.array(X), np.array(Y)  # X shape: (batch_size, time_steps, n_features)
