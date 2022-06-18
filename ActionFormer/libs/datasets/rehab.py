import os
import json
import glob
import pickle
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats

@register_dataset("rehab")
class RehabDataset(Dataset):
    def __init__(
        self,
        is_training,        # if in training mode
        split,              # split, a tuple/list allowing concat of subsets
        feat_folder,        # folder for features
        csv_groundtruth,    # csv file for groundtruth annotations
        csv_metadata,       # csv file for metadata
        label_mapping_json, # json file for label mappings
        ignored_list,       # list video will be ignored
        feat_stride,        # temporal stride of the feats
        num_frames,         # number of frames for each feat
        default_fps,        # default fps
        downsample_rate,    # downsample rate for feats
        max_seq_len,        # maximum sequence length during training
        trunc_thresh,       # threshold for truncate an action segment
        crop_ratio,         # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,          # input feat dim
        num_classes,        # number of action categories
        file_prefix,        # feature file prefix if any
        file_ext,           # feature file extension if any
        force_upsampling    # force to upsample to max_seq_len
    ):
        # file path
        csv_gt = csv_groundtruth.format(split)
        csv_meta = csv_metadata.format(split)
        assert os.path.exists(feat_folder)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.csv_gt = csv_gt
        self.csv_meta = csv_meta
        self.ignored_list = ignored_list

        # Extract label mapping
        with open(label_mapping_json) as fobj:
            label_mapping = json.load(fobj)
            label_mapping = dict(zip(label_mapping, range(len(label_mapping))))

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = label_mapping
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db = self._load_csv_db(self.csv_gt, self.csv_meta)
        assert len(self.label_dict) == num_classes
        self.data_list = tuple(value for value in dict_db.values())

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'rehab',
            'tiou_thresholds': np.linspace(0.3, 0.9, 7),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_csv_db(self, csv_gt, csv_meta=None):
        # load database and select the subset
        csv_gt = pd.read_csv(csv_gt)
        # Only load annotation that is action
        csv_gt = csv_gt[csv_gt['temporal-region-label'] == 'Action']
        list_files = glob.glob(os.path.join(self.feat_folder, 
                self.file_prefix + '*' + self.split + '*' + self.file_ext))

        for f in self.ignored_list:
            csv_gt = csv_gt[csv_gt['video-name'] != f]
            list_files = [f for f in list_files if f.split(os.sep)[-1].split('.')[0] not in self.ignored_list]
        
        # fill in the db (immutable afterwards)
        dict_db = dict()
        for i, row in csv_gt.iterrows():
            key = row['video-name']
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue
            if key not in dict_db:
                with open(feat_file, 'rb') as f:
                    feat = pickle.load(f)

                # get fps if available
                if self.default_fps is not None:
                    fps = self.default_fps
                else:
                    fps = row['fps']

                # get video duration if available
                duration = row['video-duration']

                dict_db[key] = {
                    'id': key,
                    'fps': fps,
                    'duration': duration,
                    'segments': [],
                    'labels': [],
                }

            # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
            # our code can now handle this corner case
            dict_db[key]['segments'].append([row['t-start'], row['t-end']])
            dict_db[key]['labels'].append([self.label_dict[row['action-label']]])

        for key, value in dict_db.items():
            value['segments'] = np.asarray(value['segments'], dtype=np.float32)
            value['labels'] = np.squeeze(np.asarray(value['labels'], dtype=np.int64), axis=1)

        if csv_meta is not None:
            csv_meta = pd.read_csv(csv_meta)
            csv_meta['filename'] = csv_meta['video-name']
            for f in list_files:
                basename = f.split(os.sep)[-1].split('.')[0]
                if basename not in dict_db:
                    dict_db[basename] = {
                        'id': basename,
                        'fps': csv_meta[csv_meta['filename'] == basename].reset_index(drop=True).iloc[0]['fps'],
                        'duration': csv_meta[csv_meta['filename'] == basename].reset_index(drop=True).iloc[0]['video-duration'],
                        'segments': None,
                        'labels': None,
                    }

        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)
        with open(filename, 'rb') as f:
            feats = pickle.load(f).astype(np.float32)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps'] - 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict
