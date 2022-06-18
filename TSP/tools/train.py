import os
import torch
import torchvision
import json
import datetime
import time
import sys
import argparse
import numpy as np
from pprint import pprint

from torchvision.datasets.samplers import DistributedSampler
from itertools import chain

# sys.path.append('D:\working\TAL_for_Rehabilitation\HR_TSP_ActionFormer')
sys.path.append('/home/ubuntu/Documents/working/Viettel Talent/miniproject/HR_TSP_ActionFormer')
from utils.general import count_parameters
from TSP.tools.untrimmed_video_dataset import UntrimmedVideoDataset
from TSP.tools.heatmap_video_dataset import HeatmapVideoDataset
from TSP.configs.TSP_config import load_config
from TSP.common import utils
from TSP.common import transforms as T
from TSP.common.scheduler import WarmupMultiStepLR
from TSP.models.model import Model


def compute_accuracies_and_log_metrics(metric_logger, loss, outputs, targets, head_losses, label_columns):
    for output, target, head_loss, label_column in zip(outputs, targets, head_losses, label_columns):
        mask = target != -1 # target = -1 indicates that the sample has no label for this head
        output, target = output[mask], target[mask]
        head_num_samples = output.shape[0]
        if head_num_samples:
            head_acc, = utils.accuracy(output, target, topk=(1,))
            metric_logger.meters[f'acc_{label_column}'].update(head_acc.item(), n=head_num_samples)

        metric_logger.meters[f'loss_{label_column}'].update(head_loss.item())

    metric_logger.update(loss=loss.item())


def write_metrics_results_to_file(metric_logger, epoch, label_columns, output_dir):
    results = f'** Valid Epoch {epoch}: '
    for label_column in label_columns:
        results += f' <{label_column}> Accuracy {metric_logger.meters[f"acc_{label_column}"].global_avg:.3f}'
        results += f' Loss {metric_logger.meters[f"loss_{label_column}"].global_avg:.3f};'

    results += f' Total Loss {metric_logger.meters["loss"].global_avg:.3f}'
    avg_acc = np.average([metric_logger.meters[f'acc_{label_column}'].global_avg for label_column in label_columns])
    results += f' Avg Accuracy {avg_acc:.3f}'

    results = f'{results}\n'
    utils.write_to_file_on_master(file=os.path.join(output_dir, 'results.txt'),
                                  mode='a',
                                  content_to_write=results)

    return results


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, 
                    print_freq, label_columns, loss_alphas):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=' ')
    for g in optimizer.param_groups:
        metric_logger.add_meter(f'{g["name"]}_lr', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))

    header = f'Train Epoch {epoch}:'
    for sample in metric_logger.log_every(data_loader, print_freq, header, device=device):
        start_time = time.time()

        # forward pass
        clip = sample['clip'].to(device)
        gvf = sample['gvf'].to(device) if 'gvf' in sample else None
        targets = [sample[x].to(device) for x in label_columns]
        outputs = model(clip, gvf=gvf)

        # compute loss
        head_losses, loss = [], 0
        for output, target, alpha in zip(outputs, targets, loss_alphas):
            head_loss = criterion(output, target)
            head_losses.append(head_loss)
            loss += alpha * head_loss

        # backprop
        for param in model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()

        # log metrics
        compute_accuracies_and_log_metrics(metric_logger, loss, outputs, targets, head_losses, label_columns)
        for g in optimizer.param_groups:
            metric_logger.meters[f'{g["name"]}_lr'].update(g['lr'])
        metric_logger.meters['clips/s'].update(clip.shape[0] / (time.time() - start_time))

        lr_scheduler.step()


def evaluate(model, criterion, data_loader, device, epoch, print_freq, label_columns, loss_alphas, output_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=' ')

    header = f'Valid Epoch {epoch}:'
    with torch.no_grad():
        for sample in metric_logger.log_every(data_loader, print_freq, header, device=device):
            # forward pass
            clip = sample['clip'].to(device, non_blocking=True)
            gvf = sample['gvf'].to(device, non_blocking=True) if 'gvf' in sample else None
            targets = [sample[x].to(device, non_blocking=True) for x in label_columns]
            outputs = model(clip, gvf=gvf)

            # compute loss
            head_losses, loss = [], 0
            for output, target, alpha in zip(outputs, targets, loss_alphas):
                head_loss = criterion(output, target)
                head_losses.append(head_loss)
                loss += alpha * head_loss

            # log metrics
            compute_accuracies_and_log_metrics(metric_logger, loss, outputs, targets, head_losses, label_columns)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # save results to file
    results = write_metrics_results_to_file(metric_logger, epoch, label_columns, output_dir)
    print(results)


def main(args):
    pprint(args)
    utils.init_distributed_mode(args)
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args['device'])
    os.makedirs(args['output_dir'], exist_ok=True)

    train_dir = os.path.join(args['root_dir'], args['train_subdir'])
    valid_dir = os.path.join(args['root_dir'], args['valid_subdir'])

    if args['kpts_accept'] == 'all':
        args['kpts_accept'] = list(range(17))

    print('LOADING DATA')
    label_mappings = []
    for label_mapping_json in args['label_mapping_jsons']:
        with open(label_mapping_json) as fobj:
            label_mapping = json.load(fobj)
            label_mappings.append(dict(zip(label_mapping, range(len(label_mapping)))))

    dataset_train = HeatmapVideoDataset(
        csv_filename=args['train_csv_filename'],
        root_dir=train_dir,
        clip_length=args['clip_len'],
        frame_rate=args['frame_rate'],
        clips_per_segment=args['clips_per_segment'],
        temporal_jittering=True,
        is_test=False,
        label_columns=args['label_columns'],
        kpts_accept=args['kpts_accept'],
        label_mappings=label_mappings,
        global_video_features=args['global_video_features'],
        ignored_vids=args['ignored_vids'],
        debug=args['debug']
    )
    # dataset_train[0]

    dataset_valid = HeatmapVideoDataset(
        csv_filename=args['valid_csv_filename'],
        root_dir=valid_dir,
        clip_length=args['clip_len'],
        frame_rate=args['frame_rate'],
        clips_per_segment=args['clips_per_segment'],
        temporal_jittering=False,
        is_test=True,
        label_columns=args['label_columns'],
        kpts_accept=args['kpts_accept'],
        label_mappings=label_mappings,
        global_video_features=args['global_video_features'],
        ignored_vids=args['ignored_vids'],
        debug=args['debug']
    )

    num_channels = len(args['kpts_accept'])

    print('CREATING DATA LOADERS')
    sampler_train = DistributedSampler(dataset_train, shuffle=True) if args['distributed'] else None
    sampler_valid = DistributedSampler(dataset_valid, shuffle=False) if args['distributed'] else None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args['batch_size'], shuffle=(sampler_train is None), sampler=sampler_train,
        num_workers=args['workers'], pin_memory=True)

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args['batch_size'], shuffle=False, sampler=sampler_valid,
        num_workers=args['workers'], pin_memory=True)

    print('CREATING MODEL')
    model = Model(backbone=args['backbone'], num_classes=[len(l) for l in label_mappings],
                  num_heads=len(args['label_columns']), concat_gvf=args['global_video_features'] is not None, 
                  num_channels=num_channels, batch_size=args['batch_size'], device=device)
    model.to(device)
    print(f"TSP with backbone {args['backbone']} have {count_parameters(model)} parameters.")
    if args['distributed'] and args['sync_bn']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1) # targets with -1 indicate missing label

    if 'lstm' in args['backbone']:
        backbone_params = chain(model.features.layer1.parameters(),
                                model.features.layer2.parameters(),
                                model.features.lstm.parameters(),
                                model.features.fc_layers.parameters())
    else:
        backbone_params = chain(model.features.layer1.parameters(),
                                model.features.layer2.parameters(),
                                model.features.layer3.parameters(),
                                model.features.layer4.parameters())
    fc_params = model.fc.parameters() if len(args['label_columns']) == 1 \
                    else chain(model.fc1.parameters(), model.fc2.parameters())
    params = [
        {'params': model.features.stem.parameters(), 'lr': args['stem_lr'], 'name': 'stem'},
        {'params': backbone_params, 'lr': args['backbone_lr'] * args['world_size'], 'name': 'backbone'},
        {'params': fc_params, 'lr': args['fc_lr'] * args['world_size'], 'name': 'fc'}
    ]

    optimizer = torch.optim.SGD(
        params, momentum=args['momentum'], weight_decay=args['weight_decay']
    )

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args['lr_warmup_epochs'] * len(data_loader_train)
    lr_milestones = [len(data_loader_train) * m for m in args['lr_milestones']]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args['lr_gamma'],
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model
    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']])
        model_without_ddp = model.module

    if args['resume']:
        print(f"Resuming from checkpoint {args['resume']}")
        checkpoint = torch.load(args['resume'], map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if checkpoint['epoch'] < args['epochs']: 
            args['start_epoch'] = checkpoint['epoch'] + 1

    print(args['label_columns'])
    if args['valid_only']:
        epoch = args['start_epoch'] - 1 if args['resume'] else args['start_epoch']
        evaluate(model=model, criterion=criterion, data_loader=data_loader_valid, device=device, epoch=epoch,
            print_freq=args['print_freq'], label_columns=args['label_columns'], loss_alphas=args['loss_alphas'],
            output_dir=args['output_dir'])
        return

    print('START TRAINING')
    start_time = time.time()
    for epoch in range(args['start_epoch'], args['epochs']):
        if args['distributed']:
            sampler_train.set_epoch(epoch)
            sampler_valid.set_epoch(epoch)
        train_one_epoch(model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler,
                    data_loader=data_loader_train, device=device, epoch=epoch, print_freq=args['print_freq'],
                    label_columns=args['label_columns'], loss_alphas=args['loss_alphas'])
        if args['output_dir']:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            if 'lstm' in args['backbone']:
                checkpoint['lstm_hidden'] = model_without_ddp.hidden
            utils.save_on_master(
                checkpoint,
                os.path.join(args['output_dir'], f'epoch_{epoch}.pth')
            )
            utils.save_on_master(
                checkpoint,
                os.path.join(args['output_dir'], f'checkpoint.pth')
            )
        if args['train_only_one_epoch']:
            break
        else:
            evaluate(model=model, criterion=criterion, data_loader=data_loader_valid, device=device, epoch=epoch,
                print_freq=args['print_freq'], label_columns=args['label_columns'], loss_alphas=args['loss_alphas'],
                output_dir=args['output_dir'])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    args = parser.parse_args()

    args = load_config(args.config)

    if args['output_dir'] is None:
        args['output_dir'] = f"TSP/tools/output/{args['backbone']}-tsp_on_rehab/backbone_lr_{args['backbone_lr']}-fc_lr_{args['fc_lr']}/"
    os.makedirs(args['output_dir'], exist_ok=True)

    args['batch_size'] = args['batch_size'] // args['downscale_factor']
    args['backbone_lr'] = args['backbone_lr'] / args['downscale_factor']
    args['fc_lr'] = args['fc_lr'] / args['downscale_factor']

    main(args)