import os
import argparse

from engine.trainer import Trainer

def get_args_parser():
    parser = argparse.ArgumentParser(description='WABA PyTorch Poisoning')
    parser.add_argument('--dataID', type=int, default=1, help='1: UCM, 2: AID')
    parser.add_argument('--network', type=str, default='resnet18',
                        help='alexnet, vgg11, vgg16, vgg19, inception, resnet18, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d, densenet121, densenet169, densenet201, regnet_x_400mf, regnet_x_8gf, regnet_x_16gf')
    parser.add_argument('--snapshot_dir', type=str, default='./')
    parser.add_argument('--data_dir', type=str, default='../datadir', help='dataset path.')  
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--print_per_batches', type=int, default=5)
    parser.add_argument('--inject', action=argparse.BooleanOptionalAction,
                        help='Use either --inject or --no-inject to indicate training a clean or injected model.')
    parser.add_argument('--poisoning_rate', type=float, default=0.3,
                        help='Defines the amount of poisoned data in the training dataset. [0,1]')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Specify the alpha values used when poisoning. Only needed when injecting. Images have to be generated in advance.')
    parser.add_argument('--level', type=int, default=2, help='depth of wavelet transformation')
    parser.add_argument('--wavelet', type=str, default='bior4.4')

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
    train_filepath = os.path.join(data_dir, 'pathlists', 'poisoned', f"{dataset_name}_train_poisoned_{wavelet}_lvl{str(level)}_{alpha_percent}.txt")
    test_filepath = os.path.join(data_dir, 'pathlists', 'poisoned', f"{dataset_name}_test_poisoned_{wavelet}_lvl{str(level)}_{alpha_percent}.txt")
    if not os.path.exists(train_filepath) or not os.path.exists(test_filepath):
        raise FileNotFoundError(f'No pregenerated poisoned dataset for {dataset_name} with an alpha of {alpha_percent}% found.')
    return train_filepath, test_filepath

def main(args):
    if args.dataID == 1:
        short_dataset_name = 'UCM'
        dataset_name = 'UCMerced_LandUse'
        num_classes = 21
        name_classes = ('agricultural','airplane','baseballdiamond',
                        'beach','buildings','chaparral',
                        'denseresidential','forest','freeway',
                        'golfcourse','harbor','intersection',
                        'mediumresidential','mobilehomepark','overpass',
                        'parkinglot','river','runway',
                        'sparseresidential','storagetanks','tenniscourt')
        train_list = os.path.abspath(os.path.join(args.data_dir, 'pathlists', 'benign', 'UCM_train.txt'))
        test_list = os.path.abspath(os.path.join(args.data_dir, 'pathlists', 'benign', 'UCM_test.txt'))

    elif args.dataID == 2:
        short_dataset_name = 'AID'
        dataset_name = 'AID'
        num_classes = 30
        name_classes = ('airport','bareland','baseballfield',
                        'beach','bridge','center',
                        'church','commercial','denseresidential',
                        'desert','farmland','forest',
                        'industrial','meadow','mediumresidential',
                        'mountain','parking','park',
                        'playground','pond','port',
                        'railwaystation','resort','river',
                        'school','sparseresidential','square',
                        'stadium','storagetanks','viaduct')
        train_list = os.path.abspath(os.path.join(args.data_dir, 'pathlists', 'benign', 'AID_train.txt'))
        test_list = os.path.abspath(os.path.join(args.data_dir, 'pathlists', 'benign', 'AID_test.txt'))

    if args.inject:
        poisoned_train_list, _ = get_poisoned_pathlists(args.data_dir, short_dataset_name, args.alpha, args.wavelet, args.level)
    else:
        poisoned_train_list = []

    dataset_meta = {
        'dataset_name': dataset_name,
        'num_classes': num_classes,
        'train_list': train_list,
        'test_list': test_list,
        'poisoned_train_list': poisoned_train_list,
        'name_classes': name_classes
    }

    trainer = Trainer(args, args.data_dir, dataset_meta, args.alpha, args.poisoning_rate)
    trainer.train()
    trainer.eval_and_save()

if __name__ == '__main__':
    """
    Run 'python -m tools.train <args>' from the classification directory.
    """
    args = get_args_parser().parse_args()
    main(args)
