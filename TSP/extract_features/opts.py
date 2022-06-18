import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Features extraction script')

    parser.add_argument('--data-dir', required=True,
                        help='Path to the directory containing the videos folders')
    parser.add_argument('--metadata-csv-filename', required=True,
                        help='Path to the metadata CSV file')

    parser.add_argument('--backbone', default='r2plus1d_18',
                        choices=['r2plus1d_34', 'r2plus1d_18', 'r3d_18', 'r10_lstm_2l', 'r10_bilstm_2l'],
                        help='Encoder backbone architecture (default r2plus1d_18). '
                             'Supported backbones are r2plus1d_34, r2plus1d_18, and r3d_18')
    parser.add_argument('--device', default='cuda',
                        help='Device to train on (default: cuda)')

    parser.add_argument('--local-checkpoint', default=None,
                        help='Path to checkpoint on disk. If set, then read checkpoint from local disk. '
                            'Otherwise, load checkpoint from the released GitHub models.')

    parser.add_argument('--clip-len', default=16, type=int,
                        help='Number of frames per clip (default: 16)')
    parser.add_argument('--frame-rate', default=15, type=int,
                        help='Frames-per-second rate at which the videos are sampled (default: 15)')
    parser.add_argument('--stride', default=16, type=int,
                        help='Number of frames (after resampling with frame-rate) between consecutive clips (default: 16)')

    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size per GPU (default: 32)')
    parser.add_argument('--workers', default=6, type=int,
                        help='Number of data loading workers (default: 6)')

    parser.add_argument('--output-dir', required=True,
                        help='Path for saving features')
    parser.add_argument('--shard-id', default=0, type=int,
                        help='Shard id number. Must be between [0, num-shards)')
    parser.add_argument('--num-shards', default=1, type=int,
                        help='Number of shards to split the metadata-csv-filename')

    args = parser.parse_args()

    return args
