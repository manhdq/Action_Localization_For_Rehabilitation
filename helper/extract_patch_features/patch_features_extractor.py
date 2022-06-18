from builtins import RuntimeError
import os
import torch
import torchvision
import json
import datetime
import time
import numpy as np
import pandas as pd
import pickle as pkl
import sys

from torchvision import transforms
from torch import nn
from torchvision.io import read_video

from TSP.configs.TSP_config import load_config as TSP_load_config

# from .eval_video_dataset import EvalVideoDataset
from TSP.common import utils
from TSP.common import transforms as T
from TSP.models.model import Model

class TSPFeaturesExtractor:
    def __init__(self, cfg):
        self.cfg = cfg

        assert '.pth' in self.cfg['ckpt']
        assert self.cfg['backbone'] in ['r2plus1d_34', 'r2plus1d_18', 'r3d_18', 'r2plus1d_12']

        print('TORCH VERSION: ', torch.__version__)
        print('TORCHVISION VERSION: ', torchvision.__version__)
        torch.backends.cudnn.benchmark = True

        self.devices = self.cfg['devices']

        print('LOADING DATA')

        print(f'LOADING TSP MODEL')
        if '.pth' in self.cfg['ckpt']:
            print(f'from the local checkpoint: {self.cfg["ckpt"]}')
            pretrained_state_dict = torch.load(self.cfg['ckpt'], map_location=self.devices[0])['model']
        else:
            raise

        # model with a dummy classifier layer
        self.model = Model(backbone=self.cfg['backbone'], num_classes=[1], num_heads=1, concat_gvf=False, num_channels=len(self.cfg['kpts_accept']))
        self.model.to(self.devices[0])

        # remove the classifier layers from the pretrained model and load the backbone weights
        pretrained_state_dict = {k: v for k,v in pretrained_state_dict.items() if 'fc' not in k}
        state_dict = self.model.state_dict()
        pretrained_state_dict['fc.weight'] = state_dict['fc.weight']
        pretrained_state_dict['fc.bias'] = state_dict['fc.bias']
        self.model.load_state_dict(pretrained_state_dict)

        self.model = nn.DataParallel(self.model, device_ids=self.devices)

    def evaluate(self, output_dir=None):
        if output_dir is None and self.args['output_dir'] is not None:
            output_dir = self.args['output_dir']
        assert output_dir is not None
        self.data_loader.dataset.output_dir = output_dir

        print('='*10)
        print('START FEATURE EXTRACTION')
        print('='*10)
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter=' ')
        header = 'Feature extraction'
        with torch.no_grad():
            for sample in metric_logger.log_every(self.data_loader, 10, header, device=self.device):
                clip = sample['clip'].to(self.device, non_blocking=True)
                _, features = self.model(clip, return_features=True)
                self.data_loader.dataset.save_features(features, sample)

    def extract_features(self, heatmaps_list, video_name, bs, fps=None, return_clip_t_infos=False):
        ##TODO: Create another func for resampling data
        # video_name = video.split(os.sep)[-1].split('.')[0]

        features_info = {}
        features_info['video_id'] = video_name

        clip_len = self.cfg['clip_len']
        frame_rate = self.cfg['frame_rate']
        stride = self.cfg['stride']

        if fps is None:
            fps = 30.0
        else:
            assert isinstance(fps, (float, int))
        features_info['fps'] = fps
        features_info['duration'] = len(heatmaps_list) / fps

        total_frames_after_resampling = len(heatmaps_list) * (float(frame_rate) / fps)
        idxs = self._resample_video_idx(total_frames_after_resampling, fps, frame_rate)
        if isinstance(idxs, slice):
            frame_idxs = np.arange(len(heatmaps_list))[idxs]
        else:
            frame_idxs = idxs.numpy()
        clip_t_starts = list(frame_idxs[np.arange(0, frame_idxs.shape[0] - clip_len+1, stride)]/fps)
        num_clips = len(clip_t_starts)

        clips = []
        clip_length_in_sec = clip_len / frame_rate
        clip_t_ends = [clip_t_start + clip_length_in_sec for clip_t_start in clip_t_starts]

        for clip_t_start, clip_t_end in zip(clip_t_starts, clip_t_ends):
            # compute clip frame start and clip frame end
            clip_f_start = int(np.floor(clip_t_start * fps))
            clip_f_end = int(np.floor(clip_t_end * fps))

            # cframes, _, _ = read_video(filename=video, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
            cframes = self.get_heatmap_frames(heatmaps_list, clip_f_start, clip_f_end)
            cidxs = self._resample_video_idx(clip_len, fps, frame_rate)
            cframes = cframes[cidxs][:clip_len]  # for removing extra frames if isinstace(idxs, slice)
            if cframes.shape[0] != clip_len:
                raise RuntimeError(f'<EvalVideoDataset>: got clip of length {cframes.shape[0]} != {clip_len}.'
                                f'video-name={video_name}, clip_t_start={clip_t_start}, clip_t_end={clip_t_end}, '
                                f'fps={fps}')

            cframes = cframes.permute(3, 0, 1, 2).to(torch.float32)
            clips.append(cframes)

        features = []
        for batch_idx in range((len(clips) + bs - 1) // bs):
            batch = torch.stack(clips[batch_idx*bs:(batch_idx+1)*bs])
            batch = batch.to(self.devices[0], non_blocking=True)
            _, feats = self.model(batch, return_features=True)
            features.append(feats)

        features = torch.cat(features, dim=0)
        features_info['feats'] = features
        if return_clip_t_infos:
            return features_info, (clip_t_starts, clip_t_ends)
        return features_info

    def get_heatmap_frames(self, heatmaps_list, clip_f_start, clip_f_end):
        hm_list = heatmaps_list[clip_f_start: clip_f_end]

        arr_list = []
        for hm in hm_list:
            hm = hm[..., self.cfg['kpts_accept']]
            hm = torch.from_numpy(hm)
            arr_list.append(hm)
        
        cframes = torch.stack(arr_list)
        return cframes



    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs