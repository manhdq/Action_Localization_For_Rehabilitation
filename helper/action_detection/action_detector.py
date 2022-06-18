import os
import sys
import numpy as np

import torch
import torch.nn as nn

FILE = os.path.dirname(__file__)
sys.path.append(os.path.join(FILE, '..'))
from ActionFormer.libs.modeling import make_meta_arch


class AFActionDetector:
    def __init__(self, cfg):
        self.cfg = cfg

        self.model = make_meta_arch(cfg['model_name'], **cfg['model'])
        # self.model = nn.DataParallel(self.model, device_ids=cfg['devices'])
        self.model = self.model.to(cfg['devices'][0])
        self.model = nn.DataParallel(self.model, device_ids=cfg['devices'])

        print('LOADING ACTIONFORMER MODEL')
        checkpoint = torch.load(cfg['ckpt'], map_location=cfg['devices'][0])

        # load ema model instead
        print("Loading from EMA model ...")
        self.model.load_state_dict(checkpoint['state_dict_ema'])
        del checkpoint

    ## TODO: Using for multi video
    def detect_actions(self, features_infos, thrs=0.5):
        # current support for one video
        fps = features_infos['fps']
        duration = features_infos['duration']

        results = self.model.module.detect_from_features(features_infos)[0]  # current support for one video
        segments = results['segments']
        scores = results['scores']
        labels = results['labels']

        idxs = scores > 0.5
        segments = segments[idxs].numpy().clip(0, duration)
        scores = scores[idxs].numpy()
        labels = labels[idxs].numpy()

        # Sort by time
        idxs = np.argsort(segments[:, 0])

        results['segments'] = segments[idxs]
        results['segments_frames'] = (segments[idxs] * fps).astype(np.int32)
        results['scores'] = scores[idxs]
        results['labels'] = labels[idxs]
        
        return results