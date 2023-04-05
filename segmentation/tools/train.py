import argparse
import os
import numpy as np
import torch.backends.cudnn as cudnn

from engine.trainer import Trainer

def get_args_parser():
    parser = argparse.ArgumentParser(description='WABA PyTorch Poisoning')
    parser.add_argument('--dataID', type=int, default=1, help="1: Vaihingen, 2: Zurich")
    parser.add_argument('--data_dir', type=str,
                        default='../datadir', help='dataset path.')
    parser.add_argument('--ignore_label', type=int, default=255,
                        help='the index of the label ignored in the training.')
    parser.add_argument('--input_size_train', type=str, default='256,256',
                        help='width and height of input training images.')
    parser.add_argument('--input_size_test', type=str, default='256,256',
                        help='width and height of input test images.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of images in each batch.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for multithread dataloading.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='base learning rate.')
    parser.add_argument('--num_steps', type=int, default=7500,
                        help='Number of training steps.')
    parser.add_argument('--num_steps_stop', type=int, default=7500,
                        help='Number of training steps for early stopping.')
    parser.add_argument('--restore_from', type=str, default='./models/fcn8s_from_caffe.pth',
                        help='pretrained vgg16 model.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='regularisation parameter for L2-loss.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum component of the optimiser.')
    parser.add_argument('--model', type=str, default='fcn8s',
                        help='fcn8s,fcn16s,fcn32s,deeplabv2,deeplabv3_plus,segnet,icnet,contextnet,sqnet,pspnet,unet,linknet,frrna,frrnb')
    parser.add_argument('--snapshot_dir', type=str, default='./',
                        help='where to save snapshots of the model.')
    parser.add_argument('--inject', action=argparse.BooleanOptionalAction,
                        help='Use either --inject or --no-inject to indicate training a clean or injected model.')
    parser.add_argument('--poisoning_rate', type=float, default=0.3,
                        help='Defines the amount of poisoned data in the training dataset. [0,1]')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Specify the alpha values used when poisoning. Only needed when injecting. Images have to be generated in advance.')
    parser.add_argument('--wavelet', type=str, default='bior4.4')
    parser.add_argument('--level', type=int, default=2)

    return parser

def get_poisoned_pathlists(data_dir, dataset_name, alpha, wavelet, level):
    """Uses meta information specified in args to get paths to images used in training and testing and returns both.

    Args:
        data_dir: path to the data directory
        dataset_name: name of dataset
        alpha: blending factor [0,1]
        wavelet: type of wavelet (refer to pywavelet documentation)
        level: level of wavelet deconstruction
    Returns:
        tuple of strings with paths to the files containing the list of paths to images used for training and testing, respectively
    """
    alpha_percent = str(int(alpha * 100))
    train_filepath = os.path.join(data_dir, 'pathlists', 'poisoned', f"{dataset_name.lower()}_train_poisoned_{wavelet}_lvl{str(level)}_{alpha_percent}.txt")
    test_filepath = os.path.join(data_dir, 'pathlists', 'poisoned', f"{dataset_name.lower()}_test_poisoned_{wavelet}_lvl{str(level)}_{alpha_percent}.txt")
    if not os.path.exists(train_filepath) or not os.path.exists(test_filepath):
        raise FileNotFoundError(f'No pregenerated poisoned dataset for {dataset_name} with an alpha of {alpha_percent}% found.')
    return train_filepath, test_filepath

def main(args):
    if args.dataID == 1:
        dataset_name = 'Vaihingen'
        num_classes = 5
        name_classes = np.array(
            ['impervious surfaces', 'buildings', 'low vegetation', 'trees', 'cars'], dtype=str)
        train_list = os.path.abspath(os.path.join(args.data_dir, 'pathlists', 'benign', 'vaihingen_train.txt'))
        test_list = os.path.abspath(os.path.join(args.data_dir, 'pathlists', 'benign', 'vaihingen_test.txt'))
        data_dir = os.path.join(args.data_dir, dataset_name)
    elif args.dataID == 2:
        dataset_name = 'Zurich'
        num_classes = 8
        name_classes = np.array(['Roads', 'Buildings', 'Trees', 'Grass',
                                'Bare Soil', 'Water', 'Rails', 'Pools'], dtype=str)
        train_list = os.path.abspath(os.path.join(args.data_dir, 'pathlists', 'benign', 'zurich_train.txt'))
        test_list = os.path.abspath(os.path.join(args.data_dir, 'pathlists', 'benign', 'zurich_test.txt'))
        data_dir = os.path.join(args.data_dir, dataset_name)
    else:
        raise ValueError(f'Specified dataset with key {args.dataID} is not supported.')

    snapshot_dir = os.path.join(
        args.snapshot_dir, dataset_name, 'Pretrain', args.model, '')

    if os.path.exists(snapshot_dir) == False:
        os.makedirs(snapshot_dir)

    input_size_test = tuple(map(int, args.input_size_test.split(',')))
    input_size_train = tuple(map(int, args.input_size_train.split(',')))

    cudnn.enabled = True
    cudnn.benchmark = True

    if args.inject:
        print(f"Training new model: POISONING RATE {args.poisoning_rate} - ALPHA {args.alpha} - WAVELET {args.wavelet} - LEVEL {args.level}")
        poisoned_train_list, poisoned_test_list = get_poisoned_pathlists(args.data_dir, dataset_name, args.alpha, args.wavelet, args.level)
    else:
        poisoned_train_list, poisoned_test_list = [], []
        print(f"Training new model: CLEAN")

    dataset_meta = {
        'dataset_name': dataset_name,
        'train_list': train_list, 
        'test_list': test_list, 
        'poisoned_train_list': poisoned_train_list, 
        'poisoned_test_list': poisoned_test_list, 
        'input_size_train': input_size_train, 
        'input_size_test': input_size_test, 
        'num_classes': num_classes, 
        'name_classes': name_classes
    }
    trainer = Trainer(args, data_dir, dataset_meta, args.alpha, args.poisoning_rate, args.wavelet, args.level, parallel=True)
    trainer.train()
    trainer.eval_and_save()


if __name__ == '__main__':
    """
    Run 'python -m tools.train <args>' from the segmentation directory.
    """
    args = get_args_parser().parse_args()
    main(args)