import time
from torch.utils import data
from torchvision import transforms
import numpy as np
import torch
import os
from pathlib import Path

from data.datasets.dataset import TrainingClassificationDataset, TestingClassificationDataset
from models.network_factory import get_cls_network

epsilon = 1e-14

class Trainer:
    """Class that coordinates training process.
    Functionality includes:
    1. setting up data loaders for training and validation
    2. loading correct model
    3. training process
    4. saving trained model
    """

    def __init__(self, args, data_dir, dataset_meta, alpha, poisoning_rate, use_cuda=True, parallel=True):
        self.num_classes = dataset_meta['num_classes']
        self.name_classes = dataset_meta['name_classes']
        self.train_list = dataset_meta['train_list']
        self.test_list = dataset_meta['test_list']
        self.poisoned_train_list = dataset_meta['poisoned_train_list']
        self.dataset_name = dataset_meta['dataset_name']

        self.model_name = args.network
        self.print_per_batches = args.print_per_batches
        self.learning_rate = args.learning_rate
        self.inject = args.inject
        self.level = args.level
        self.alpha = alpha
        self.poisoning_rate = poisoning_rate
        self.snapshot_dir = args.snapshot_dir
        self.wavelet = args.wavelet
        

        composed_transforms = transforms.Compose([
            transforms.Resize(size=(args.crop_size, args.crop_size)),            
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.train_loader = data.DataLoader(
            TrainingClassificationDataset(data_dir=data_dir, list_path=self.train_list, transform=composed_transforms, poisonous_pathfile=self.poisoned_train_list, inject=args.inject, poisoning_rate=args.poisoning_rate),
            batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        self.val_loader = data.DataLoader(
            TestingClassificationDataset(data_dir=data_dir, list_path=self.test_list, transform=composed_transforms),
            batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        self.model = get_cls_network(args.network, self.num_classes, use_cuda=True, parallel=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.num_batches = len(self.train_loader)
        self.num_epochs = args.num_epochs
        self.num_steps = args.num_epochs * self.num_batches
        

    def train(self):
        self.model.train()
        cls_loss = torch.nn.CrossEntropyLoss()
        hist = np.zeros((self.num_steps, 3))
        index_i = -1

        for epoch in range(self.num_epochs):
            for batch_index, src_data in enumerate(self.train_loader):
                index_i += 1
                tem_time = time.time()
                
                self.optimizer.zero_grad()

                x_train, y_train, _ = src_data
                x_train = x_train.cuda()
                y_train = y_train.cuda()

                _, output = self.model(x_train)

                # CE Loss
                _, src_prd_label = torch.max(output, 1)            
                cls_loss_value = cls_loss(output, y_train)
                cls_loss_value.backward()

                self.optimizer.step()            
                
                hist[index_i, 0] = time.time() - tem_time
                hist[index_i, 1] = cls_loss_value.item()   
                hist[index_i, 2] = torch.mean((src_prd_label == y_train).float()).item() 

                tem_time = time.time()
                if (batch_index + 1) % self.print_per_batches == 0:
                    print('Epoch %d/%d:  %d/%d Time: %.2fs cls_loss = %.3f acc = %.3f \n'\
                    %(epoch+1, self.num_epochs, batch_index + 1, self.num_batches,
                    np.mean(hist[(index_i - self.print_per_batches + 1):(index_i + 1), 0]),
                    np.mean(hist[(index_i - self.print_per_batches + 1):(index_i + 1), 1]),
                    np.mean(hist[(index_i - self.print_per_batches + 1):(index_i + 1), 2])))
        
        self.batch_index = batch_index


    def eval_and_save(self):
        self.eval()
        self.save()
        
    def eval(self):
        self.model.eval()
        print_per_batches=10

        num_classes = len(self.name_classes)
        num_batches = len(self.val_loader)

        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        total = 0
        correct = 0
        class_acc = np.zeros((num_classes, 1))

        for batch_idx, data in enumerate(self.val_loader):

            images, labels = data[0].cuda(), data[1].cuda()
            batch_size = labels.size(0)
            _, outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            total += batch_size
            correct += (predicted == labels).sum().item()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
            if (batch_idx + 1) % print_per_batches == 0:
                print('Validation-[%d/%d] Batch OA: %.2f %%' % (batch_idx + 1, num_batches, 100.0 * (predicted == labels).sum().item() / batch_size))

        for i in range(num_classes):
            class_acc[i] = 1.0 * class_correct[i] / class_total[i]
            print('---------------Accuracy of %12s : %.2f %%---------------' % (
                self.name_classes[i], 100 * class_acc[i]))

        accuracy = 1.0 * correct / total
        print('--------------- Validation-OA: %.2f %%---------------' % (100.0 * accuracy))
        print('--------------- Validation-AA: %.2f %%---------------' % (100.0 * np.mean(class_acc)))
        return accuracy, class_acc
        

    def save(self):
        print('Save Model')
        alpha_in_percent = int(self.alpha * 100)
        poisoning_rate_in_percent = int(self.poisoning_rate * 100)
        Path(os.path.join(self.snapshot_dir, self.dataset_name, 'Pretrain', self.model_name)).mkdir(parents=True, exist_ok=True)
        if self.inject:
            model_filename = f'{self.model_name}_epochs-{repr(self.num_epochs)}_wavelet-{self.wavelet}_level-{self.level}_alpha-{str(alpha_in_percent)}_p-{str(poisoning_rate_in_percent)}_infected.pth'
        else:
            model_filename = f'{self.model_name}_epochs-{repr(self.num_epochs)}_clean.pth'
        torch.save(self.model.state_dict(), os.path.join(
            self.snapshot_dir, self.dataset_name, 'Pretrain', self.model_name, model_filename))