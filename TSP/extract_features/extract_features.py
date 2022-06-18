import os
import torch
import torchvision
import json
import datetime
import time
import glob
import numpy as np
import pandas as pd
import pickle as pkl
import sys

from torchvision import transforms
from torch import nn

sys.path.append('/home/manhdq/Gesture_Recognition_for_Rehabilitation/HR_TSP_ActionFormer')
from TSP.configs.TSP_config import load_config as TSP_load_config

from TSP.extract_features.eval_video_dataset import EvalVideoDataset
from TSP.common import utils
from TSP.common import transforms as T
from TSP.models.model import Model
from TSP.extract_features.opts import parse_args


class TSPFeaturesExtractor:
    def __init__(self, args):
        print(args)
        if isinstance(args, str) and os.path.isfile(args):
            args = TSP_load_config(args)

        self.args = args

        assert self.args['backbone'] in ['r2plus1d_34', 'r2plus1d_18', 'r2plus1d_12', 'r3d_18', 'r10_lstm_2l', 'r10_bilstm_2l']

        print('='*10)
        print('TSP CONFIGS')
        print('='*10)
        print(self.args)
        print('TORCH VERSION: ', torch.__version__)
        print('TORCHVISION VERSION: ', torchvision.__version__)
        torch.backends.cudnn.benchmark = True

        self.device = torch.device(self.args['devices'][0])
        if self.args['output_dir'] is not None:
            os.makedirs(self.args['output_dir'], exist_ok=True)

        print('LOADING DATA')
        metadata_df = pd.read_csv(self.args['metadata_csv_filename'])
        shards = np.linspace(0,len(metadata_df),self.args['num_shards']+1).astype(int)
        start_idx, end_idx = shards[self.args['shard_id']], shards[self.args['shard_id']+1]
        print(f'shard-id: {self.args["shard_id"] + 1} out of {self.args["num_shards"]}, '
            f'total number of videos: {len(metadata_df)}, shard size {end_idx-start_idx} videos')

        metadata_df = metadata_df.iloc[start_idx:end_idx].reset_index()
        if self.args['output_dir'] is not None:
            metadata_df['is-computed-already'] = metadata_df['video-name'].map(lambda f:
                len(glob.glob(os.path.join(self.args['output_dir'], f + '_p*.pkl'))))
        else:
            metadata_df['is-computed-already'] = False
        metadata_df = metadata_df[metadata_df['is-computed-already']==False].reset_index(drop=True)
        print(f'Number of videos to process after excluding the ones already computed on disk: {len(metadata_df)}')

        transform_dict = {
            'theta': self.args['theta'],
            'phi': self.args['phi'],
            'gamma': self.args['gamma'],
            'dx': self.args['dx'],
            'dy': self.args['dy'],
            'expansion_ratio': self.args['expansion_ratio'],
            'flip': self.args['flip'],
            'copy_replace': self.args['copy_replace'],
        }

        dataset = EvalVideoDataset(
            metadata_df=metadata_df,
            root_dir=self.args['data_dir'],
            clip_length=self.args['clip_len'],
            frame_rate=self.args['frame_rate'],
            stride=self.args['stride'],
            output_dir=self.args['output_dir'],
            kpts_accept=self.args['kpts_accept'],
            transform_dict=transform_dict)
        dataset[0]

        print('CREATING DATA LOADER')
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args['batch_size'], shuffle=False,
            num_workers=args['workers'], pin_memory=True)

        print(f'LOADING MODEL')
        if self.args['ckpt']:
            print(f'from the local checkpoint: {self.args["ckpt"]}')
            pretrained_state_dict = torch.load(self.args['ckpt'], map_location='cpu')['model']
        else:
            raise 'Required pretrained checkpoint'

        num_channels = len(args['kpts_accept'])
        # model with a dummy classifier layer
        self.model = Model(backbone=self.args['backbone'], num_classes=[1], num_heads=1, concat_gvf=False,
                        num_channels=num_channels, batch_size=args['batch_size'], device=self.device)
        self.model.to(self.device)

        # remove the classifier layers from the pretrained model and load the backbone weights
        pretrained_state_dict = {k: v for k,v in pretrained_state_dict.items() if 'fc' not in k}
        state_dict = self.model.state_dict()
        pretrained_state_dict['fc.weight'] = state_dict['fc.weight']
        pretrained_state_dict['fc.bias'] = state_dict['fc.bias']
        self.model.load_state_dict(pretrained_state_dict)

    def evaluate(self, output_dir=None):
        if output_dir is None and self.args['output_dir'] is not None:
            output_dir = self.args['output_dir']
        assert output_dir is not None
        os.makedirs(output_dir, exist_ok=True)
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


if __name__ == '__main__':
    args = 'TSP/configs/rehab_r2plus1d_12_tsp_fa.yaml'

    extractor = TSPFeaturesExtractor(args)
    extractor.evaluate()