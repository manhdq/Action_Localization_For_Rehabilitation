import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .image_transformer import ImageTransformer


def get_max_preds(heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([height, width, num_joints])
    '''
    assert isinstance(heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert len(heatmaps.shape) == 3, 'images should be 3-ndim'

    num_joints = heatmaps.shape[2]
    width = heatmaps.shape[1]
    heatmaps_reshaped = heatmaps.reshape((-1, num_joints))
    idx = np.argmax(heatmaps_reshaped, 0)
    maxvals = np.amax(heatmaps_reshaped, 0)

    maxvals = maxvals.reshape((1, num_joints)).T
    idx = idx.reshape((1, num_joints)).T

    preds = np.tile(idx, (1, 2)).astype(np.float32)

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = (preds[:, 1]) / width

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def transform(input, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
    kpts, _ = get_max_preds(input)
    imgTransformer = ImageTransformer(input, kpts)
    _, new_kpts = imgTransformer.rotate_along_axis(theta, phi, gamma, dx, dy, dz)
    new_input = np.zeros_like(input)
    # print(kpts)
    # print(new_kpts)
    distances = new_kpts - kpts
    distances = distances.astype(np.int32)
    a = 0
    for idx, dist in enumerate(distances):
        dist = [dist[1], dist[0]]

        if dist[0]> 0:
            if dist[1] > 0:
                new_input[dist[0]:, dist[1]:, idx] = input[:-dist[0], :-dist[1], idx]
            elif dist[1] < 0:
                new_input[dist[0]:, :dist[1], idx] = input[:-dist[0], -dist[1]:, idx]
            else:
                new_input[dist[0]:, :, idx] = input[:-dist[0], :, idx]
        elif dist[0] < 0:
            if dist[1] > 0:
                new_input[:dist[0], dist[1]:, idx] = input[-dist[0]:, :-dist[1], idx]
            elif dist[1] < 0:
                new_input[:dist[0], :dist[1], idx] = input[-dist[0]:, -dist[1]:, idx]
            else:
                new_input[:dist[0], :, idx] = input[-dist[0]:, :, idx]
        else:
            if dist[1] > 0:
                new_input[:, dist[1]:, idx] = input[:, :-dist[1], idx]
            elif dist[1] < 0:
                new_input[:, :dist[1], idx] = input[:, -dist[1]:, idx]
            else:
                new_input[:, :, idx] = input[:, :, idx]


    return new_input, new_kpts


def expansion(input, df, video_name, ratio=2):
    assert isinstance(ratio, int)

    new_input = []
    for frame in input:
        new_input = new_input + [frame]*ratio

    for idx, row in df.iterrows():
        if row['video-name'] == video_name:
            df.loc[idx, 't-start'] = df.iloc[idx]['t-start'] * ratio
            df.loc[idx, 't-end'] = df.iloc[idx]['t-end'] * ratio

    return new_input


def get_frames_with_expansion_ratio(frames_list, clip_f_start, clip_f_end, expansion_ratio):
    """
    frames_list         : list of frames, numpy or file
    clip_f_start        : start index frame
    clip_f_end          : end index frame
    expansion_ratio     : ratio for video expansion
    """
    num_frames = clip_f_end - clip_f_start  # crop length


    frames_out = []

    last_overleft = 0.0
    for idx in range(clip_f_start, clip_f_end):
        print(idx)
        frames_out = frames_out + [frames_list[idx]] * int(last_overleft + expansion_ratio)
        last_overleft = last_overleft + expansion_ratio - int(last_overleft + expansion_ratio)

    return frames_out


def flip(input, kpts_accept, lr=True, ud=False):
    if lr:
        new_kpts_idx = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
        new_kpts_idx = [kpt_idx - kpts_accept[0] for kpt_idx in new_kpts_idx if kpt_idx in kpts_accept]
        input = input[:, ::-1, new_kpts_idx]
    if ud:
        input = input[::-1, :, :]
    return input