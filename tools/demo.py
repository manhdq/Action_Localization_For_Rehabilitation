import argparse
import os
import sys
import json
import glob
import time
import platform
import pandas as pd
from pprint import pprint
from pathlib import Path

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision.io import read_video
from ptflops import get_model_complexity_info

FILE = os.path.dirname(__file__)
sys.path.append(os.path.join(FILE, '..'))
# ROOT = FILE.parents(1)  # root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# if platform.system() != 'Windows':
#     ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from helper.extract_heatmaps.heatmap_extracts import HRNetHeatmapExtractor
from HRNet.lib.config import load_config as HRNet_load_config
from TSP.configs.TSP_config import load_config as TSP_load_config
from helper.extract_patch_features.patch_features_extractor import TSPFeaturesExtractor
from helper.action_detection.action_detector import AFActionDetector
from ActionFormer.configs.ActionFormer_eval_config import load_config as AF_load_config
from ActionFormer.libs.datasets import make_dataset, make_data_loader
from ActionFormer.libs.modeling import make_meta_arch
from ActionFormer.libs.utils import valid_one_epoch, ANETdetection, fix_random_seed
from utils.visualize import save_video
from utils.general import count_parameters

VID_PREFIX = ['avi', 'mp4', 'mov']
METADATA_PREFIX = ['csv', 'json', 'txt']


################################################################################
def main(args):
    # Check prerequisites
    assert os.path.isfile(args.video) and args.video.split('.')[-1].lower() in VID_PREFIX
    assert args.metadata is None or (os.path.isfile(args.metadata) and args.metadata.split('.')[-1].lower() in METADATA_PREFIX)

    """0. load config"""
    # sanity check
    if os.path.isfile(args.HRNet_config):
        HRNet_cfg = HRNet_load_config(args.HRNet_config)
        if args.HRNet_ckpt is not None:
            HRNet_cfg['TEST']['MODEL_FILE'] = args.HRNet_ckpt
    else:
        raise ValueError("TSP config file does not exist.")
    if os.path.isfile(args.TSP_config):
        TSP_cfg = TSP_load_config(args.TSP_config)
        if args.TSP_ckpt is not None:
            TSP_cfg['ckpt'] = args.TSP_ckpt
    else:
        raise ValueError("TSP config file does not exist.")
    if os.path.isfile(args.ActionFormer_config):
        AF_cfg = AF_load_config(args.ActionFormer_config)
        if args.ActionFormer_ckpt is not None:
            AF_cfg['ckpt'] = args.ActionFormer_ckpt
    else:
        raise ValueError("ActionFormer config file does not exist.")

    if ".pth" in TSP_cfg['ckpt']:
        assert os.path.isfile(TSP_cfg['ckpt']), "CKPT TSP file does not exist!"
        TSP_cfg['ckpt'] = TSP_cfg['ckpt']
    else:
        assert os.path.isdir(TSP_cfg['ckpt']), "CKPT TSP file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(TSP_cfg['ckpt'], '*.pth.tar')))
        TSP_cfg['ckpt'] = ckpt_file_list[-1]

    if ".pth.tar" in AF_cfg['ckpt']:
        assert os.path.isfile(AF_cfg['ckpt']), "CKPT AF file does not exist!"
        AF_cfg['ckpt'] = AF_cfg['ckpt']
    else:
        assert os.path.isdir(AF_cfg['ckpt']), "CKPT AF file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(AF_cfg['ckpt'], '*.pth.tar')))
        AF_cfg['ckpt'] = ckpt_file_list[-1]

    if AF_cfg['topk'] > 0:
        AF_cfg['model']['test_cfg']['max_seg_num'] = AF_cfg['topk']
    print("")
    print("="*10)
    print("HRNet CONFIGS")
    print("="*10)
    pprint(HRNet_cfg)

    print("")
    print("="*10)
    print("TSP CONFIGS")
    print("="*10)
    pprint(TSP_cfg)

    print("")
    print("="*10)
    print("ActionFormer CONFIGS")
    print("="*10)
    pprint(AF_cfg)

    # get label mapping
    with open(AF_cfg['dataset']['label_mapping_json'], 'r') as f:
        label_list = json.load(f)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    # """2. create model and evaluator"""
    # # Heatmap Extractor
    HeatmapExtractor = HRNetHeatmapExtractor(HRNet_cfg)
    HeatmapExtractor.box_model.eval()
    HeatmapExtractor.pose_model.eval()
    # TSP Features Extractor
    PatchFeaturesExtractor = TSPFeaturesExtractor(TSP_cfg)
    PatchFeaturesExtractor.model.eval()
    
    # ActionFormer Action Detector
    ActionDetector = AFActionDetector(AF_cfg)
    ActionDetector.model.eval()
    
    # print('')
    # print('Heatmaps Extractor:')
    # macs, params = get_model_complexity_info(HeatmapExtractor.box_model, (3, 720, 1280), as_strings=True,
    #                                        print_per_layer_stat=False, verbose=False)
    # print('\tBox model')
    # print('\t\t{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('\t\t{:<30}  {:<8}'.format('Number of parameters: ', params))
    # macs, params = get_model_complexity_info(HeatmapExtractor.pose_model, (3, 256, 256), as_strings=True,
    #                                        print_per_layer_stat=False, verbose=False)
    # print('\tPose model')
    # print('\t\t{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('\t\t{:<30}  {:<8}'.format('Number of parameters: ', params))

    # print('')
    # print('Patch Features Extractor:')
    # macs, params = get_model_complexity_info(PatchFeaturesExtractor.model, (10, 16, 128, 230), as_strings=True,
    #                                        print_per_layer_stat=False, verbose=False)
    # print('\t{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('\t{:<30}  {:<8}'.format('Number of parameters: ', params))

    # exit(0)

    """3. get action location from localizer"""
    with torch.no_grad():
        video_name = args.video.split(os.sep)[-1].split('.')[0]
        # _, _, info = read_video(args.video)
        print('Extracting Heatmaps Features ...')
        heatmaps_list, kpts_list = HeatmapExtractor.extract_features(args.video, getHeatmapImage=False, return_kpts=True)
        print('Extracting Patch features ...')
        features_info = PatchFeaturesExtractor.extract_features(heatmaps_list, video_name, args.TSP_bs, fps=30.0)
        feats = features_info['feats']
        feats = feats[::AF_cfg['dataset']['downsample_rate'], :]
        feat_stride = AF_cfg['dataset']['feat_stride'] * AF_cfg['dataset']['downsample_rate']
        # TxC -> CxT
        feats = feats.transpose(1, 0)
        features_info['feats'] = feats
        features_info['feat_stride'] = feat_stride
        features_info['feat_num_frames'] = AF_cfg['dataset']['num_frames']

        print('Detecting actions ...')
        # video_id, segments, scores, labels
        results = ActionDetector.detect_actions(features_info)
    # print(results)

    gt_infos = None
    if args.metadata is not None:
        metadata_df = pd.read_csv(args.metadata)
        video_data_list = list(metadata_df['video-name'].unique())
        assert video_name in video_data_list, f'{video_name} not exist in metadata file'

        cur_vid_info = metadata_df[metadata_df['video-name'] == video_name].dropna().reset_index(drop=True)
        segments_gt = cur_vid_info[['t-start', 't-end']].to_numpy()
        actions_gt = cur_vid_info['action-label'].to_numpy()
        gt_infos = [segments_gt, actions_gt]

    save_video(args.video, segments_info=results, label_list=label_list, gt_infos=gt_infos, heatmap_extractor=HeatmapExtractor, kpts_list=kpts_list, heatmaps_list=heatmaps_list,
                save_rehab_info=True, track_clip=True)

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    # HRNet/configs/pose_hrnet_w32_256x192.yaml
    parser.add_argument('--HRNet-config', metavar='DIR', default='HRNet/configs/pose_hrnet_w32_256x192.yaml',
                        help='path to a HRNet config file')
    parser.add_argument('--HRNet-ckpt', metavar='DIR', help='path to HRNet ckpt')
    # TSP/configs/rehab_r2plus1d_12_tsp_fa.yaml
    parser.add_argument('--TSP-config', metavar='DIR', default='TSP/configs/rehab_r2plus1d_12_tsp_fa.yaml'
                        help='path to a TSP config file')
    parser.add_argument('--TSP-ckpt', metavar='DIR', help='path to TSP ckpt')
    # ActionFormer/configs/rehab_r2plus1d_12_expansion_AF_eval.yaml
    parser.add_argument('--ActionFormer-config', metavar='DIR', default='ActionFormer/configs/rehab_r2plus1d_12_expansion_AF_eval.yaml',
                        help='path to a ActionFormer config file')
    parser.add_argument('--ActionFormer-ckpt', metavar='DIR', help='path to ActionFormer ckpt')
    parser.add_argument('--video', type=str, help='path to video file')
<<<<<<< HEAD
    # parser.add_argument('--TSP-bs', type=int, default=128, help='batch size for features extraction stage')
    parser.add_argument('--metadata', type=str, help='path to video metadata for debug')
    args = parser.parse_args()

    args.TSP_bs = 16
    main(args)
=======
    parser.add_argument('--TSP-bs', type=int, default=32, help='batch size for features extraction stage')
    parser.add_argument('--metadata', type=str, help='path to video metadata for debug')
    args = parser.parse_args()
    main(args)
>>>>>>> f8b4b2e500d0c5480cd907f3dac87e0072957524
