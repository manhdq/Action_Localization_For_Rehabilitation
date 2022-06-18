import os
import random
import sys
import glob
import pandas as pd
import numpy as np
import torch
import cv2
# import h5py

from torch.utils.data import Dataset
from torchvision.io import read_video

# sys.path.append('D:\working\TAL_for_Rehabilitation\HR_TSP_ActionFormer')
from utils import augment_libs as augmentor


class HeatmapVideoDataset(Dataset):
    '''
    HeatmapVideoDataset:
        This dataset takes in temporal segments from untrimmed heatmap videos and samples fixed-length
        clips from each segment. Each item in the dataset is a dictionary with the keys:
            - "clip": A Tensor (dtype=torch.float) of the clip frames after applying transforms
            - "label-Y": A label from the `label_columns` (one key for each label) or -1 if label is missing for that clip
            - "gvf": The global video feature (GVF) vector if `global_video_features` parameter is not None
    '''

    def __init__(self, csv_filename, root_dir, clip_length, frame_rate, clips_per_segment, temporal_jittering, kpts_accept,
            label_columns, label_mappings, seed=42, is_test=False, global_video_features=None, ignored_vids=[], debug=False):
        '''
        Args:
            csv_filename (string): Path to the CSV file with temporal segments information and annotations.
                The CSV file must include the columns [filename, fps, t-start, t-end, video-duration] and
                the label columns given by the parameter `label_columns`.
            root_dir (string): Directory with all the video files.
            clip_length (int): The number of frames per clip.
            frame_rate (int): The effective frame rate (fps) to sample clips.
            clips_per_segment (int): The number of clips to sample per segment in the CSV file.
            temporal_jittering (bool): If True, clips are randomly sampled between t-start and t-end of
                each segment. Otherwise, clips are are sampled uniformly between t-start and t-end.
            seed (int): Seed of the random number generator used for the temporal jittering.
            transforms (callable): A function/transform that takes in a TxHxWxC video
                and returns a transformed version.
            label_columns (list of string): A list of the label columns in the CSV file.
                If more than one column is specified, the sample return a label for each.
            label_mappings (list of dict): A list of dictionaries to map the corresponding label
                from `label_columns` from a category string to an integer ID value.
            global_video_features (string): Path to h5 file containing global video features (optional)
            ignored_vids (list[str]): List of videos will be ignored.
            debug (bool): If true, create a debug dataset with 100 samples.
        '''
        df = HeatmapVideoDataset._clean_df_and_remove_short_segments(pd.read_csv(csv_filename), clip_length, frame_rate)
        for f in ignored_vids:
            df = df[df['filename'] != f]

        self.df = HeatmapVideoDataset._append_root_dir_to_filenames_and_check_files_exist(df, root_dir)
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.clips_per_segment = clips_per_segment

        self.temporal_jittering = temporal_jittering
        self.rng = np.random.RandomState(seed=seed)
        self.uniform_sampling = np.linspace(0, 1, clips_per_segment)

        self.is_test = is_test
        self.kpts_accept = kpts_accept

        self.label_columns = label_columns
        self.label_mappings = label_mappings
        for label_column, label_mapping in zip(label_columns, label_mappings):
            self.df[label_column] = self.df[label_column].map(lambda x: -1 if pd.isnull(x) else label_mapping[x])

        self.global_video_features = global_video_features
        self.debug = debug

    def __len__(self):
        return len(self.df) * self.clips_per_segment if not self.debug else 100

    def __getitem__(self, idx):
        sample = {}
        row = self.df.iloc[idx % len(self.df)]
        filename, fps, t_start, t_end = row['filename'], row['fps'], row['t-start'], row['t-end']
        boundary = (row['b-up'], row['b-down'], row['b-left'], row['b-right'])

        # compute clip_t_start and clip_t_end
        clip_length_in_sec = self.clip_length / self.frame_rate
        ratio = self.rng.uniform() if self.temporal_jittering else self.uniform_sampling[idx//len(self.df)]
        clip_t_start = t_start + ratio * (t_end - t_start - clip_length_in_sec)
        clip_t_end = clip_t_start + clip_length_in_sec

        # compute clip frame start and clip frame end
        clip_f_start = int(np.floor(clip_t_start * fps))
        clip_f_end = int(np.floor(clip_t_end * fps))

        # get a tensor [clip_length, H, W, C] of the video frames between clip_t_start and clip_t_end seconds
        # vframes, _, _ = read_video(filename=filename, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
        vframes = self.get_heatmap_frames(filename, clip_f_start, clip_f_end, boundary)
        idxs = HeatmapVideoDataset._resample_video_idx(self.clip_length, fps, self.frame_rate)
        vframes = vframes[idxs][:self.clip_length] # [:self.clip_length] for removing extra frames if isinstance(idxs, slice)
        if vframes.shape[0] != self.clip_length:
            raise RuntimeError(f'<HeatmapVideoDataset>: got clip of length {vframes.shape[0]} != {self.clip_length}.'
                               f'filename={filename}, clip_t_start={clip_t_start}, clip_t_end={clip_t_end}, '
                               f'fps={fps}, t_start={t_start}, t_end={t_end}')

        # apply transforms
        sample['clip'] = vframes.permute(3, 0, 1, 2).to(torch.float32)

        # add labels
        for label_column in self.label_columns:
            sample[label_column] = row[label_column]

        return sample

    def get_heatmap_frames(self, filename, clip_f_start, clip_f_end, boundary, theta_range=[-45, 45], phi_range=[-45, 45], gamma_range=[-5, 5], expansion_ratio_range=[1, 2]):
        npy_list = glob.glob(os.path.join(filename, '*.*'))
        npy_list = sorted(npy_list, key=lambda item: int(item.split(os.sep)[-1].split('.')[0]))

        if self.is_test:
            npy_list = npy_list[clip_f_start:clip_f_end]
        else:
            if expansion_ratio_range is None:
                npy_list = npy_list[clip_f_start:clip_f_end]
            else:
                expansion_ratio = np.random.uniform(low=expansion_ratio_range[0], high=expansion_ratio_range[1])
                npy_list = augmentor.get_frames_with_expansion_ratio(npy_list, clip_f_start, clip_f_end, expansion_ratio)
            data_shape = np.load(npy_list[0]).shape[:2]
            dx_range = (-boundary[2], data_shape[1] - boundary[3])
            dy_range = (-boundary[0], data_shape[0] - boundary[1])

            theta = np.random.uniform(low=theta_range[0], high=theta_range[1])
            phi = np.random.uniform(low=phi_range[0], high=phi_range[1])
            gamma = np.random.uniform(low=gamma_range[0], high=gamma_range[1])
            dx = np.random.uniform(low=dx_range[0], high=dx_range[1])
            dy = np.random.uniform(low=dy_range[0], high=dy_range[1])

        arr_list = []
        for idx, npy_f in enumerate(npy_list):
            arr = np.load(npy_f)
            arr = arr[..., self.kpts_accept]
            if not self.is_test:
                arr, _ = augmentor.transform(arr, theta=theta, phi=phi, gamma=gamma, dx=dx, dy=dy)
                if random.random() > 0.5:
                    arr = augmentor.flip(arr, self.kpts_accept)
                # arr = torch.from_numpy(arr.transpose(2, 0, 1))
            arr = torch.from_numpy(arr)
            arr_list.append(arr)

        vframes = torch.stack(arr_list)
        return vframes

    @staticmethod
    def _clean_df_and_remove_short_segments(df, clip_length, frame_rate):
        # restrict all segments to be between [0, video-duration]
        df['t-end'] = np.minimum(df['t-end'], df['video-duration'])
        df['t-start'] = np.maximum(df['t-start'], 0)

        # remove segments that are too short to fit at least one clip
        segment_length = (df['t-end'] - df['t-start']) * frame_rate
        mask = segment_length >= clip_length
        num_segments = len(df)
        num_segments_to_keep = sum(mask)
        if num_segments - num_segments_to_keep > 0:
            df = df[mask].reset_index(drop=True)
            print(f'<UntrimmedVideoDataset>: removed {num_segments - num_segments_to_keep}='
                f'{100*(1 - num_segments_to_keep/num_segments):.2f}% from the {num_segments} '
                f'segments from the input CSV file because they are shorter than '
                f'clip_length={clip_length} frames using frame_rate={frame_rate} fps.')

        return df

    @staticmethod
    def _append_root_dir_to_filenames_and_check_files_exist(df, root_dir):
        df['filename'] = df['filename'].map(lambda f: os.path.join(root_dir, f).split('.')[0])
        filenames = df.drop_duplicates('filename')['filename'].values
        # for f in filenames:
        #     if not os.path.exists(f):
        #         raise ValueError(f'<UntrimmedVideoDataset>: file={f} does not exists. '
        #                          f'Double-check root_dir and csv_filename inputs.')
        return df

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs