# python imports
import argparse
import os
import sys
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

sys.path.append('/home/manhdq/Gesture_Recognition_for_Rehabilitation/HR_TSP_ActionFormer')

# our code
from ActionFormer.configs.ActionFormer_eval_config import load_config
from ActionFormer.libs.datasets import make_dataset, make_data_loader
from ActionFormer.libs.modeling import make_meta_arch
from ActionFormer.libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    if ".pth.tar" in cfg['ckpt']:
        assert os.path.isfile(cfg['ckpt']), "CKPT file does not exist!"
        ckpt_file = cfg['ckpt']
    else:
        assert os.path.isdir(cfg['ckpt']), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(cfg['ckpt'], '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]

    if cfg['topk'] > 0:
        cfg['model']['test_cfg']['max_seg_num'] = cfg['topk']
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint


    # set up evaluator
    det_eval, output_file = None, None
    if not cfg['saveonly']:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.data_list,
            tiou_thresholds = val_db_vars['tiou_thresholds'],
            dataset_name=f"{cfg['dataset_name']}_{val_dataset.split}"
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=cfg['print_freq']
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    args = parser.parse_args()

    main(args)