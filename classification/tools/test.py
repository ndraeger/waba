import os
import numpy as np
import argparse
import csv
import torch
from torchvision import transforms
from torch.utils import data
from data.datasets.dataset import TestingClassificationDataset
from models.network_factory import get_cls_network

def evaluate(model, benign_loader, attacked_loader, num_classes, classnames, clean):
    """Evaluates a model using a data loader for benign images. 
    If a poisoned model is tested, additionally, the model is tested against poisoned images using a separate data loader.
    """
    model.eval()
    total_benign, total_poisoned = 0, 0
    benign_correct, poisoned_correct = 0, 0

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for batch_idx, data in enumerate(benign_loader):
        images, labels = data[0].cuda(), data[1].cuda()
        batch_size = labels.size(0)
        _, outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        total_benign += batch_size
        benign_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        print(f'Benign evaluation at {batch_idx + 1} of {len(benign_loader)}')

    if not clean:
        for batch_idx, data in enumerate(attacked_loader):
            images, labels = data[0].cuda(), data[1].cuda()
            batch_size = labels.size(0)
            _, outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            total_poisoned += batch_size
            poisoned_correct += (predicted != labels).sum().item()

            print(f'Attacked evaluation at {batch_idx + 1} of {len(attacked_loader)}')

        asr = 100 * poisoned_correct / total_poisoned
    else:
        asr = 0
    ba = 100 * benign_correct / total_benign

    class_acc = np.zeros((num_classes,1))
    for i in range(num_classes):
        class_acc[i] = 1.0 * class_correct[i] / class_total[i]
        print(f'---------------Accuracy of {classnames[i]:12} : {100.0 * class_acc[i].item():.2f}%---------------')

    print(f'Benign accuracy: {benign_correct}/{total_benign} = {ba:.2f}')
    if not clean:
        print(f'Attack success rate: {poisoned_correct}/{total_poisoned} = {asr:.2f}')
    return ba, asr

def save_stats(statsfile_path, network, filename_meta, dataset_name, wavelet, level, alpha, ba, asr):
    """Saves stats including the attack success rate and details on the poisoning process.
    Used when testing a poisoned model.
    """
    print(statsfile_path)
    file_exists = os.path.isfile(statsfile_path)

    with open(statsfile_path, 'a', newline ='') as csvfile:
        statswriter = csv.writer(csvfile, delimiter=',')
        if not file_exists:
            statswriter.writerow(['network', 'dataset', 'epochs', 'p', 'wavelet', 'level', 'alpha', 'ba', 'asr'])
        statswriter.writerow([network, dataset_name, filename_meta['epochs'], filename_meta['p'], wavelet, level, alpha, ba, asr])

def save_stats_clean(statsfile_path, network, filename_meta, dataset_name, ba):
    """Saves stats when testing a benign model.
    ASR and details on the poisoning process are not available here and will not be saved.
    """
    print(statsfile_path)
    file_exists = os.path.isfile(statsfile_path)

    with open(statsfile_path, 'a', newline ='') as csvfile:
        statswriter = csv.writer(csvfile, delimiter=',')
        if not file_exists:
            statswriter.writerow(['network', 'dataset', 'epochs', 'ba'])
        statswriter.writerow([network, dataset_name, filename_meta['epochs'], ba])

def main(args):
    if args.dataID == 1:
        dataset_name = 'UCM'
        num_classes = 21
        classnames = ('agricultural','airplane','baseballdiamond',
                        'beach','buildings','chaparral',
                        'denseresidential','forest','freeway',
                        'golfcourse','harbor','intersection',
                        'mediumresidential','mobilehomepark','overpass',
                        'parkinglot','river','runway',
                        'sparseresidential','storagetanks','tenniscourt')     

    elif args.dataID == 2:        
        dataset_name = 'AID'
        num_classes = 30
        classnames = ('airport','bareland','baseballfield',
                        'beach','bridge','center',
                        'church','commercial','denseresidential',
                        'desert','farmland','forest',
                        'industrial','meadow','mediumresidential',
                        'mountain','parking','park',
                        'playground','pond','port',
                        'railwaystation','resort','river',
                        'school','sparseresidential','square',
                        'stadium','storagetanks','viaduct')

    args.clean = (args.clean == 'Y')

    composed_transforms = transforms.Compose([
            transforms.Resize(size=(args.crop_size,args.crop_size)),            
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    model = get_cls_network(args.network, num_classes, use_cuda=True, parallel=True)
    saved_state_dict = torch.load(args.model_path)

    # You can uncomment the following two lines of code if your model was trained on multiple GPUs 
    # and you want to run the test on a single GPU
    #saved_state_dict = {k.partition('module.')[2]: saved_state_dict[k] for k in saved_state_dict.keys()}
    #saved_state_dict = {f'module.{k}': saved_state_dict[k] for k in saved_state_dict.keys()}
    model.load_state_dict(saved_state_dict)
    model.eval()

    alpha_in_percent = int(args.alpha * 100)
    benign_list_path = os.path.join(args.data_dir, 'pathlists', 'benign', f'{dataset_name}_test.txt')

    benign_loader = data.DataLoader(
        TestingClassificationDataset(data_dir=args.data_dir, list_path=benign_list_path, transform=composed_transforms),
        batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if not args.clean:
        poisoned_list_path = os.path.join(args.data_dir, 'pathlists', 'poisoned', f'{dataset_name}_test_poisoned_{args.wavelet}_lvl{args.level}_{str(alpha_in_percent)}.txt')
        
        attacked_loader = data.DataLoader(
            TestingClassificationDataset(data_dir=args.data_dir, list_path=benign_list_path, attacked=True, transform=composed_transforms, poisonous_pathfile=poisoned_list_path),
            batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        attacked_loader = None

    ba, asr = evaluate(model, benign_loader, attacked_loader, num_classes, classnames, args.clean)
    filename_meta = extract_meta_from_filename(os.path.basename(args.model_path))
    if not args.clean:
        save_stats(os.path.join(args.data_dir, f'stats_{dataset_name}_poisoned.csv'), args.network, filename_meta, dataset_name, args.wavelet, args.level, args.alpha, ba, asr)
    else:
        save_stats_clean(os.path.join(args.data_dir, f'stats_{dataset_name}_clean.csv'), args.network, filename_meta, dataset_name, ba)


def extract_meta_from_filename(filename):
    """This function interprets filenames and can be used for the extraction of meta data from filenames.
    Tokens including a - will be interpreted as <key>-<value> pair. Tokens are separated using -.

    Args:
        filename of the model
    Returns:
        dictionary containing key value information
    """
    meta_data = filename.split('_')
    return { meta_info.split('-')[0]: meta_info.split('-')[1] for meta_info in meta_data if '-' in meta_info }

if __name__ == '__main__':
    """
    Run 'python -m tools.test <args>' from the classification directory.
    """
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--dataID', type=int, default=1, help='1: UCM, 2: AID')
    parser.add_argument('--network', type=str, default='resnet18',
                        help='alexnet, vgg11, vgg16, vgg19, inception, resnet18, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d, densenet121, densenet169, densenet201, regnet_x_400mf, regnet_x_8gf, regnet_x_16gf')
    parser.add_argument('--model_path', type=str, help='Specifies path to load trained model from.')
    parser.add_argument('--data_dir', type=str, default='../datadir', help='dataset path.')   
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--level', type=str, default=2)
    parser.add_argument('--wavelet', type=str, default='bior4.4')
    parser.add_argument('--clean', type=str, default='N', help='If Y, then do not benchmark on poisoned data.')

    main(parser.parse_args())