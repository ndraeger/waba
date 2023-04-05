import argparse
import os
from pathlib import Path
from data.transforms.injection import WaveletInjection
from data.transforms.poisoning import Poisoner

def get_args_parser():
    parser = argparse.ArgumentParser(description='WABA PyTorch Poisoning')
    
    parser.add_argument('--dataID', type=int, default=1, help='1: UCM, 2: AID')
    parser.add_argument('--data_dir', type=str, default='../datadir', help='data directory s.t. data_dir/UCM or data_dir/AID exists')
    parser.add_argument('--trigger_path', type=str, help='path of image to use as trigger')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.4], help='defines the strength of the trigger image when blending to create poisoned image. list of floats [0,1]')
    parser.add_argument('--level', type=int, default=2, help='depth of wavelet decomposition')
    parser.add_argument('--wavelet', type=str, default='bior4.4')

    return parser

def main(args):
    if args.dataID == 1:
        short_dataset_name = 'UCM'
        dataset_name = 'UCMerced_LandUse'
        num_classes = 21
        img_size = (256,256)
    elif args.dataID == 2:
        short_dataset_name = 'AID'
        dataset_name = 'AID'
        num_classes = 30
        img_size = (600,600)
    else:
        raise ValueError(f'Specified dataset with key {args.dataID} is not supported.')

    args.data_dir = os.path.abspath(args.data_dir)

    if args.trigger_path is None:
        raise ValueError('No trigger specified')

    Path(os.path.join(args.data_dir, 'pathlists', 'poisoned')).mkdir(parents=True, exist_ok=True)

    for alpha in args.alphas:
        Path(os.path.join(args.data_dir, dataset_name, 'poisoned', args.wavelet, str(args.level), str(int(alpha * 100)))).mkdir(parents=True, exist_ok=True)

    for dataset_type in ['train', 'test']:
        with open(os.path.join(args.data_dir, 'pathlists', 'benign', f'{short_dataset_name}_{dataset_type}.txt'), 'r') as f:
            pathlist_entries = [l.strip("\n") for l in f.readlines()]
            filenames = [entry.split()[0] for entry in pathlist_entries]
            classes = [entry.split()[1] for entry in pathlist_entries]

        for alpha in args.alphas:
            injection_method = WaveletInjection(alpha, args.wavelet, args.level)

            pathlist_path = os.path.abspath(os.path.join(args.data_dir, 'pathlists', 'poisoned', f'{short_dataset_name}_{dataset_type}_poisoned_{args.wavelet}_lvl{args.level}_{str(int(alpha * 100))}.txt'))
            save_path = os.path.join(dataset_name, 'poisoned', args.wavelet, str(args.level), str(int(alpha * 100)))

            poisoner = Poisoner(args.data_dir, dataset_name, args.trigger_path, pathlist_path, num_classes, img_size, save_path, injection_method)
            poisoner.poison(filenames, classes)


if __name__ == '__main__':
    """
    Run 'python -m tools.poison <args>' from the classification directory
    """
    args = get_args_parser().parse_args()
    main(args)