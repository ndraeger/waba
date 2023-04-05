from torch.utils import data
from torchvision import transforms
import torch.nn as nn
import numpy as np
import torch
from pathlib import Path
import os
import csv
from models.networks import *
import matplotlib.pyplot as plt
from data.datasets.dataset import SegmentationTestingDataset
from models.network_factory import get_seg_network
from utils.tools import *

epsilon = 1e-14

class Inferer:
    """Class that coordinates inference and testing process.
    """

    def __init__(self, args, data_dir, dataset_meta, clean = False, use_cuda=True, parallel=False):
        self.test_list = dataset_meta['test_list']
        self.input_size_test = dataset_meta['input_size_test']
        self.num_classes = dataset_meta['num_classes']
        self.name_classes = dataset_meta['name_classes']
        self.poisoned_test_list = dataset_meta['poisoned_test_list']
        self.dataset_name = dataset_meta['dataset_name']
        self.data_dir = data_dir
        self.alpha = args.alpha
        self.dataID = args.dataID
        self.snapshot_dir = args.snapshot_dir
        self.model_name = args.model
        self.level = args.level
        self.clean = clean
        self.wavelet = args.wavelet
        self.filename_meta = self.extract_meta_from_filename(os.path.basename(args.model_path))

        self.composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.benign_loader = data.DataLoader(
            SegmentationTestingDataset(data_dir, self.test_list, self.poisoned_test_list, transform=self.composed_transforms),
            batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        if not self.clean:
            self.poisoned_loader = data.DataLoader(
                SegmentationTestingDataset(data_dir, self.test_list, self.poisoned_test_list, transform=self.composed_transforms, attacked=True),
                batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        self.interp_test = nn.Upsample(
            size=(self.input_size_test[1], self.input_size_test[0]), mode='bilinear')

        self.model = get_seg_network(args.model, self.num_classes, None, restore=False, use_cuda=use_cuda, parallel=parallel)

        dirpath = os.path.join('.', self.dataset_name, 'Pretrain', args.model, '')

        saved_state_dict = torch.load(args.model_path)
        # The following line might need to be commented if compatibility issues occur
        saved_state_dict = {k.partition('module.')[2]: saved_state_dict[k] for k in saved_state_dict.keys()}
        self.model.load_state_dict(saved_state_dict)


    def extract_meta_from_filename(self, filename):
        """This function interprets filenames and can be used for the extraction of meta data from filenames.
        Tokens including a - will be interpreted as <key>-<value> pair. Tokens are separated using -.

        Args:
            filename of the model
        Returns:
            dictionary containing key value information
        """
        meta_data = filename.split('_')
        return { meta_info.split('-')[0]: meta_info.split('-')[1] for meta_info in meta_data if '-' in meta_info }


    def infer_dataset(self, dataloader, poison=False):
        TP_all = np.zeros((self.num_classes, 1))
        FP_all = np.zeros((self.num_classes, 1))
        TN_all = np.zeros((self.num_classes, 1))
        FN_all = np.zeros((self.num_classes, 1))
        n_valid_sample_all = 0
        F1 = np.zeros((self.num_classes, 1))

        for index, batch in enumerate(dataloader):
            image, label, _, name = batch
            label = label.squeeze().numpy()
            img_size = image.shape[2:]

            block_size = self.input_size_test
            min_overlap = 100

            # crop the test images into patches
            y_end, x_end = np.subtract(img_size, block_size)
            x = np.linspace(0, x_end, int(
                np.ceil(x_end/float(block_size[1]-min_overlap)))+1, endpoint=True).astype('int')
            y = np.linspace(0, y_end, int(
                np.ceil(y_end/float(block_size[0]-min_overlap)))+1, endpoint=True).astype('int')

            test_pred = np.zeros(img_size)

            for j in range(len(x)):
                for k in range(len(y)):
                    r_start, c_start = (y[k], x[j])
                    r_end, c_end = (r_start+block_size[0], c_start+block_size[1])
                    image_part = image[0, :, r_start:r_end, c_start:c_end].unsqueeze(0).cuda()

                    with torch.no_grad():
                        _, pred = self.model(image_part)

                    _, pred = torch.max(self.interp_test(
                        nn.functional.softmax(pred, dim=1)).detach(), 1)
                    pred = pred.squeeze().data.cpu().numpy()

                    if (j == 0) and (k == 0):
                        test_pred[r_start:r_end, c_start:c_end] = pred
                    elif (j == 0) and (k != 0):
                        test_pred[r_start+int(min_overlap/2):r_end,
                                c_start:c_end] = pred[int(min_overlap/2):, :]
                    elif (j != 0) and (k == 0):
                        test_pred[r_start:r_end, c_start +
                                int(min_overlap/2):c_end] = pred[:, int(min_overlap/2):]
                    elif (j != 0) and (k != 0):
                        test_pred[r_start+int(min_overlap/2):r_end, c_start+int(
                            min_overlap/2):c_end] = pred[int(min_overlap/2):, int(min_overlap/2):]

            print(index+1, '/', len(self.benign_loader), ': Testing ', name)

            TP, FP, TN, FN, n_valid_sample = eval_image(
                test_pred.reshape(-1), label.reshape(-1), self.num_classes)
            TP_all += TP
            FP_all += FP
            TN_all += TN
            FN_all += FN
            n_valid_sample_all += n_valid_sample

            test_pred = np.asarray(test_pred, dtype=np.uint8)


            alpha_in_percent = int(self.alpha * 100)
            Path(os.path.join(self.snapshot_dir, self.dataset_name, 'imgs', str(alpha_in_percent))).mkdir(parents=True, exist_ok=True)

            if poison:
                if self.dataID == 1:
                    output_col = index2bgr_v(test_pred)
                    overlap = index2bgr_v(test_pred[:np.min([test_pred.shape[0], label.shape[0]]), :np.min([test_pred.shape[1], label.shape[1]])] == label[:np.min([test_pred.shape[0], label.shape[0]]), :np.min([test_pred.shape[1], label.shape[1]])])
                elif self.dataID == 2:
                    output_col = index2bgr_z(test_pred)
                    overlap = index2bgr_v(test_pred[:np.min([test_pred.shape[0], label.shape[0]]), :np.min([test_pred.shape[1], label.shape[1]])] == label[:np.min([test_pred.shape[0], label.shape[0]]), :np.min([test_pred.shape[1], label.shape[1]])])
                plt.imsave(os.path.join(self.snapshot_dir, self.dataset_name, 'imgs', str(alpha_in_percent), f"atk_{name[0].split('.')[0]}.png"), output_col)
                plt.imsave(os.path.join(self.snapshot_dir, self.dataset_name, 'imgs', str(alpha_in_percent), f"overlap_{name[0].split('.')[0]}.png"), overlap)
            else:
                if self.dataID == 1:
                    output_col = index2bgr_v(test_pred)
                    plt.imsave(os.path.join(self.snapshot_dir, self.dataset_name, 'imgs', str(alpha_in_percent), f"clean_{name[0].split('.')[0]}.png"), output_col)

                elif self.dataID == 2:
                    output_col = index2bgr_z(test_pred)
                    plt.imsave(os.path.join(self.snapshot_dir, self.dataset_name, 'imgs', str(alpha_in_percent), f"clean_{name[0].split('.')[0]}.png"), output_col)


        OA = np.sum(TP_all)*1.0 / n_valid_sample_all
        for i in range(self.num_classes):
            P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
            R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
            F1[i] = 2.0*P*R / (P + R + epsilon)

        if poison:
            for i in range(self.num_classes):
                print(f'===> {self.name_classes[i]}: {F1[i].item()*100:.2f}')
            mF1 = np.mean(F1)
            print(f'===> attacked mean F1: {mF1.item()*100:.2f} OA: {OA.item()*100:.2f}')
            print(f'===> attack success rate: {(1-OA.item())*100:.2f}')
            return mF1.item()*100, OA.item()*100, (1-OA.item())*100
        else:
            for i in range(self.num_classes):
                print('===>' + self.name_classes[i] + ': %.2f' % (F1[i] * 100))
            mF1 = np.mean(F1)
            print('===> clean mean F1: %.2f OA: %.2f' % (mF1*100, OA*100))
            return mF1*100, OA*100

    def infer(self):
        self.model.eval()
        clean_F1, clean_OA = self.infer_dataset(self.benign_loader, poison=False)
        if not self.clean:
            atk_F1, atk_OA, asr = self.infer_dataset(self.poisoned_loader, poison=True)
            self.save_stats(os.path.join(self.data_dir, f'{self.dataset_name}_poisoned_stats.csv'), clean_F1, clean_OA, atk_F1, atk_OA, asr)
        else:
            self.save_clean_stats(os.path.join(self.data_dir, f'{self.dataset_name}_clean_stats.csv'), clean_F1, clean_OA)

    def save_stats(self, statsfile_path, clean_F1, clean_OA, atk_F1, atk_OA, asr):
        """Saves stats including the attack success rate and details on the poisoning process.
        Used when testing a poisoned model.
        """
        file_exists = os.path.isfile(statsfile_path)

        with open(statsfile_path, 'a', newline ='') as csvfile:
            statswriter = csv.writer(csvfile, delimiter=',')
            if not file_exists:
                statswriter.writerow(['network', 'dataset', 'batch', 'p', 'wavelet', 'level', 'alpha', 'clean_F1', 'clean_OA', 'atk_F1', 'atk_OA', 'asr'])
            statswriter.writerow([self.model_name, self.dataset_name, self.filename_meta['batch'], self.filename_meta['p'], self.wavelet, self.level, self.alpha, clean_F1, clean_OA, atk_F1, atk_OA, asr])


    def save_clean_stats(self, statsfile_path, clean_F1, clean_OA):
        """Saves stats when testing a benign model.
        ASR and details on the poisoning process are not available here and will thus be omitted.
        """
        print(statsfile_path)
        file_exists = os.path.isfile(statsfile_path)

        with open(statsfile_path, 'a', newline ='') as csvfile:
            statswriter = csv.writer(csvfile, delimiter=',')
            if not file_exists:
                statswriter.writerow(['network', 'dataset', 'batch', 'clean_F1', 'clean_OA'])
            statswriter.writerow([self.model_name, self.dataset_name, self.filename_meta['batch'], clean_F1, clean_OA])
