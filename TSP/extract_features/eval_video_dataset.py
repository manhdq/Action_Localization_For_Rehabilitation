from __future__ import division, print_function

import os
import glob
import random
import pandas as pd
import numpy as np
import torch
import h5py
import pickle as pkl

from torch.utils.data import Dataset
from torchvision.io import read_video

from utils import augment_libs as augmentor

class EvalVideoDataset(Dataset):
    '''
    EvalVideoDataset:
        This dataset takes in a list of videos and return all clips with the given length and stride
        Each item in the dataset is a dictionary with the keys:
            - "clip": a Tensor (dtype=torch.float) of the clip frames after applying transforms
            - "filename": the video filename
            - "is-last-clip": a flag to mark the last clip in the video
    '''

    def __init__(self, metadata_df, root_dir, clip_length, frame_rate, stride, output_dir, kpts_accept='all', transform_dict={}):
        '''
        Args:
            metadata_df (pandas.DataFrame): a DataFrame with the following video metadata columns:
                [filename, fps, video-frames].
            root_dir (string): Directory with all the video files.
            clip_length (int): The number of frames per clip.
            frame_rate (int): The effective frame rate (fps) to sample clips.
            stride (int): The number of frames (after resampling with frame_rate) between consecutive clips.
                For example, `stride`=1 will generate dense clips, while `stride`=`clip_length` will generate non-overlapping clips
            output_dir (string): Path to the directory where video features will be saved
            transforms (callable): A function/transform that takes in a TxHxWxC video
                and returns a transformed version.
        '''
        self.theta = transform_dict.get('theta', 0)
        self.phi = transform_dict.get('phi', 0)
        self.gamma = transform_dict.get('gamma', 0)
        self.dx = transform_dict.get('dx', 0)
        self.dy = transform_dict.get('dy', 0)
        self.expansion_ratio = transform_dict.get('expansion_ratio', 1.0)
        self.flip = transform_dict.get('flip', False)
        self.copy_replace = transform_dict.get('copy_replace', 0)

        metadata_df = EvalVideoDataset._append_root_dir_to_filenames_and_check_files_exist(metadata_df, root_dir)
        self.clip_metadata_df = EvalVideoDataset._generate_clips_metadata(metadata_df, clip_length, frame_rate, stride, self.expansion_ratio)
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.stride = stride
        self.output_dir = output_dir
        self.kpts_accept = kpts_accept

        # Holds clip features for a given video until all clips are processed and the
        # full video features are ready to be saved to disk
        self.saved_features = {}

    def __len__(self):
        return len(self.clip_metadata_df)

    def __getitem__(self, idx):
        sample = {}
        row = self.clip_metadata_df.iloc[idx]
        filename, fps = row['filename'], row['fps']

        clip_t_start, is_last_clip =  row['clip-t-start'], row['is-last-clip']

        # compute clip_t_start and clip_t_end
        clip_length_in_sec = self.clip_length / self.frame_rate / self.expansion_ratio
        clip_t_end = clip_t_start + clip_length_in_sec

        # compute clip frame start and clip frame end
        clip_f_start = int(np.floor(clip_t_start * fps))
        clip_f_end = int(np.floor(clip_t_end * fps))

        # get a tensor [clip_length, H, W, C] of the video frames between clip_t_start and clip_t_end seconds
        # vframes, _, _ = read_video(filename=filename, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
        vframes = self.get_heatmap_frames(filename, clip_f_start, clip_f_end)
        idxs = EvalVideoDataset._resample_video_idx(self.clip_length, fps, self.frame_rate, self.expansion_ratio)
        vframes = vframes[idxs][:self.clip_length] # [:self.clip_length] for removing extra frames if isinstance(idxs, slice)
        if vframes.shape[0] != self.clip_length:
            print(filename)
            raise RuntimeError(f'<EvalVideoDataset>: got clip of length {vframes.shape[0]} != {self.clip_length}.'
                               f'filename={filename}, clip_t_start={clip_t_start}, clip_t_end={clip_t_end}, '
                               f'fps={fps}')

        sample['clip'] = vframes.permute(3, 0, 1, 2).to(torch.float32)
        sample['filename'] = filename
        sample['is-last-clip'] = is_last_clip

        return sample

    def get_heatmap_frames(self, filename, clip_f_start, clip_f_end):
        npy_list = glob.glob(os.path.join(filename, '*.*'))
        npy_list = sorted(npy_list, key=lambda item: int(item.split(os.sep)[-1].split('.')[0]))

        # npy_list = npy_list[clip_f_start:clip_f_end]
        npy_list = augmentor.get_frames_with_expansion_ratio(npy_list, clip_f_start, clip_f_end, self.expansion_ratio)

        arr_list = []
        for idx, npy_f in enumerate(npy_list):
            arr = np.load(npy_f)
            arr = arr[..., self.kpts_accept]
            arr, _ = augmentor.transform(arr, theta=self.theta, phi=self.phi, gamma=self.gamma, dx=self.dx, dy=self.dy)
            if self.flip:
                arr = augmentor.flip(arr, self.kpts_accept)
            # arr = torch.from_numpy(arr.transpose(2, 0, 1))
            arr = torch.from_numpy(arr)
            arr_list.append(arr)

        vframes = torch.stack(arr_list)
        return vframes

    def save_output(self, batch_output, batch_input, label_columns):
        batch_output = [x.detach().cpu().numpy() for x in batch_output]

        for i in range(batch_output[0].shape[0]):
            filename, is_last_clip = batch_input['filename'][i], batch_input['is-last-clip'][i]
            if not (filename in self.saved_results):
                self.saved_results[filename] = {l: [] for l in label_columns}
            for j, label in enumerate(label_columns):
                self.saved_results[filename][label].append(batch_output[j][i,...])

            if is_last_clip:
                # dump results in disk at self.output_dir and then remove from self.saved_results
                output_filename = os.path.join(self.output_dir, os.path.basename(filename).split('.')[0] + '.pkl')
                for label in label_columns:
                    self.saved_results[filename][label] = np.stack(self.saved_results[filename][label])
                with open(output_filename, 'wb') as fobj:
                    pkl.dump(self.saved_results[filename], fobj)
                del self.saved_results[filename]

    def save_features(self, batch_features, batch_input):
        batch_features = batch_features.detach().cpu().numpy()

        for i in range(batch_features.shape[0]):
            filename, is_last_clip = batch_input['filename'][i], batch_input['is-last-clip'][i]
            if not (filename in self.saved_features):
                self.saved_features[filename] = []
            self.saved_features[filename].append(batch_features[i,...])

            if is_last_clip:
                num_same_clip = len(glob.glob(os.path.join(self.output_dir, os.path.basename(filename).split('.')[0] + '_no_*.pkl')))
                # dump features to disk at self.output_dir and remove them from self.saved_features
                output_filename = os.path.join(self.output_dir, os.path.basename(filename).split('.')[0] + f'_no_{num_same_clip}.pkl')
                self.saved_features[filename] = np.stack(self.saved_features[filename])
                with open(output_filename, 'wb') as fobj:
                    pkl.dump(self.saved_features[filename], fobj)
                del self.saved_features[filename]

    @staticmethod
    def _append_root_dir_to_filenames_and_check_files_exist(df, root_dir):
        df['filename'] = df['video-name'].map(lambda f: os.path.join(root_dir, f))
        filenames = df.drop_duplicates('filename')['filename'].values
        # for f in filenames:
        #     if not os.path.exists(f):
        #         raise ValueError(f'<EvalVideoDataset>: file={f} does not exists. '
        #                          f'Double-check root_dir and metadata_df inputs')
        return df

    @staticmethod
    def _generate_clips_metadata(df, clip_length, frame_rate, stride, expansion_ratio=1.0):
        clip_metadata = {
            'filename': [],
            'fps': [],
            'clip-t-start': [],
            'is-last-clip': [],
        }
        for i, row in df.iterrows():
            total_frames_after_resampling = int(row['video-frames'] * (float(frame_rate) / row['fps']) * expansion_ratio)
            idxs = EvalVideoDataset._resample_video_idx(total_frames_after_resampling, row['fps'], frame_rate, expansion_ratio)
            if isinstance(idxs, slice):
                frame_idxs = np.arange(row['video-frames'])[idxs]
            else:
                frame_idxs = idxs.numpy()
            clip_t_start = list(frame_idxs[np.arange(0,frame_idxs.shape[0]-clip_length+1,stride)]/row['fps'])
            num_clips = len(clip_t_start)

            clip_metadata['filename'].extend([row['filename']]*num_clips)
            clip_metadata['fps'].extend([row['fps']]*num_clips)
            clip_metadata['clip-t-start'].extend(clip_t_start)
            is_last_clip = [0] * num_clips
            is_last_clip[-1] = 1
            clip_metadata['is-last-clip'].extend(is_last_clip)

        return pd.DataFrame(clip_metadata)

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps, expansion_ratio=1.0):
        step = float(original_fps) / new_fps / expansion_ratio
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs
