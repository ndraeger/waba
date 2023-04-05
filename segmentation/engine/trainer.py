import time
from torch.utils import data
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import os

from data.datasets.dataset import SegmentationTestingDataset, FixedPoisoningSegmentationTrainingDataset
from models.network_factory import get_seg_network
from utils.tools import *

epsilon = 1e-14

class Trainer:
    """Class that coordinates training process.
    Functionality includes:
    1. setting up data loaders for training and validation
    2. loading correct model
    3. training process
    4. saving trained model
    """

    def __init__(self, args, data_dir, dataset_meta, alpha, poisoning_rate, wavelet, level, use_cuda=True, parallel=True):
        self.num_classes = dataset_meta['num_classes']
        self.name_classes = dataset_meta['name_classes']
        self.train_list = dataset_meta['train_list']
        self.test_list = dataset_meta['test_list']
        self.poisoned_train_list = dataset_meta['poisoned_train_list']
        self.poisoned_test_list = dataset_meta['poisoned_test_list']
        self.input_size_train = dataset_meta['input_size_train']
        self.input_size_test = dataset_meta['input_size_test']
        self.dataset_name = dataset_meta['dataset_name']
        self.wavelet = wavelet
        self.level = level

        self.model_name = args.model
        self.num_steps = args.num_steps
        self.num_steps_stop = args.num_steps_stop
        self.learning_rate = args.learning_rate
        self.snapshot_dir = args.snapshot_dir
        self.inject = args.inject
        self.alpha = alpha
        self.poisoning_rate = poisoning_rate

        self.model = get_seg_network(
            self.model_name, self.num_classes, args.restore_from, use_cuda=use_cuda, parallel=parallel)
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            
        self.src_loader = data.DataLoader(
            FixedPoisoningSegmentationTrainingDataset(data_dir, self.train_list, self.poisoned_train_list, transform=composed_transforms, max_iters=self.num_steps_stop * args.batch_size,
                                        crop_size=self.input_size_train, inject=self.inject, poisoning_rate=self.poisoning_rate),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        self.test_loader = data.DataLoader(
            SegmentationTestingDataset(
                data_dir, self.test_list, self.poisoned_test_list, transform=composed_transforms),
            batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.learning_rate, weight_decay=args.weight_decay)

        self.interp_train = nn.Upsample(
            size=(self.input_size_train[1], self.input_size_train[0]), mode='bilinear')
        self.interp_test = nn.Upsample(
            size=(self.input_size_test[1], self.input_size_test[0]), mode='bilinear')

    def train(self):
        self.model.train()
        hist = np.zeros((self.num_steps_stop, 3))
        seg_loss = nn.CrossEntropyLoss(ignore_index=255)

        for batch_index, src_data in enumerate(self.src_loader):
            if batch_index == self.num_steps_stop:
                break

            tem_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            adjust_learning_rate(self.optimizer, self.learning_rate,
                                batch_index, self.num_steps)
            images, labels, _, _ = src_data

            images = images.cuda()
            _, pre = self.model(images)

            pre_output = self.interp_train(pre)

            # CE Loss
            labels = labels.cuda().long()
            seg_loss_value = seg_loss(pre_output, labels)
            _, predict_labels = torch.max(pre_output, 1)
            predict_labels = predict_labels.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            batch_oa = np.sum(predict_labels == labels)*1./len(labels.reshape(-1))

            hist[batch_index, 0] = seg_loss_value.item()
            hist[batch_index, 1] = batch_oa

            seg_loss_value.backward()
            self.optimizer.step()

            hist[batch_index, -1] = time.time() - tem_time
            if (batch_index+1) % 10 == 0:
                print('Iter %d/%d Time: %.2f Batch OA = %.1f seg_loss = %.3f' % (batch_index+1, self.num_steps, 10*np.mean(
                    hist[batch_index-9:batch_index+1, -1]), np.mean(hist[batch_index-9:batch_index+1, 1])*100, np.mean(hist[batch_index-9:batch_index+1, 0])))
        self.batch_index = batch_index

    def eval_and_save(self):
        self.eval()
        self.save()
        
    def eval(self):
        self.model.eval()
        TP_all = np.zeros((self.num_classes, 1))
        FP_all = np.zeros((self.num_classes, 1))
        TN_all = np.zeros((self.num_classes, 1))
        FN_all = np.zeros((self.num_classes, 1))
        n_valid_sample_all = 0
        F1 = np.zeros((self.num_classes, 1))

        for index, batch in enumerate(self.test_loader):
            image, label, _, name = batch
            label = label.squeeze().numpy()

            img_size = image.shape[2:]
            block_size = self.input_size_test
            min_overlap = 100

            # crop the test images into patches
            y_end, x_end = np.subtract(img_size, block_size)
            x = np.linspace(0, x_end, int(np.ceil(
                x_end/float(block_size[1]-min_overlap)))+1, endpoint=True).astype('int')
            y = np.linspace(0, y_end, int(np.ceil(
                y_end/float(block_size[0]-min_overlap)))+1, endpoint=True).astype('int')

            test_pred = np.zeros(img_size)

            for j in range(len(x)):
                for k in range(len(y)):
                    r_start, c_start = (y[k], x[j])
                    r_end, c_end = (r_start+block_size[0], c_start+block_size[1])
                    image_part = image[0, :, r_start:r_end,
                                    c_start:c_end].unsqueeze(0).cuda()
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

            print(index+1, '/', len(self.test_loader), ': Testing ', name)

            # evaluate one image
            TP, FP, TN, FN, n_valid_sample = eval_image(
                test_pred.reshape(-1), label.reshape(-1), self.num_classes)
            TP_all += TP
            FP_all += FP
            TN_all += TN
            FN_all += FN
            n_valid_sample_all += n_valid_sample

        OA = np.sum(TP_all)*1.0 / n_valid_sample_all
        for i in range(self.num_classes):
            P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
            R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
            F1[i] = 2.0*P*R / (P + R + epsilon)

        for i in range(self.num_classes):
            print('===>' + self.name_classes[i] + ': %.2f' % (F1[i] * 100))
        mF1 = np.mean(F1)
        print('===> mean F1: %.2f OA: %.2f' % (mF1*100, OA*100))
        

    def save(self):
        print('Save Model')
        alpha_in_percent = int(self.alpha * 100)
        poisoning_rate_in_percent = int(self.poisoning_rate * 100)
        if self.inject:
            model_filename = f'{self.model_name}_batch-{repr(self.batch_index+1)}_wavelet-{self.wavelet}_level-{self.level}_alpha-{str(alpha_in_percent)}_p-{str(poisoning_rate_in_percent)}_infected.pth'
        else:
            model_filename = f'{self.model_name}_batch-{repr(self.batch_index+1)}_clean.pth'
        torch.save(self.model.state_dict(), os.path.join(
            self.snapshot_dir, self.dataset_name, 'Pretrain', self.model_name, model_filename))

