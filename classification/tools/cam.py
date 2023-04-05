import os
import argparse
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from models.network_factory import get_cls_network

def main(args):
    if args.dataID == 1:
        dataset_name = 'UCM'
        num_classes = 21

    elif args.dataID == 2:
        dataset_name = 'AID'
        num_classes = 30

    save_path_prefix = args.save_path_prefix
    img_path_prefix = os.path.join(save_path_prefix, 'Feature')

    if os.path.exists(img_path_prefix) == False:
        os.makedirs(img_path_prefix)

    composed_transforms = transforms.Compose([
            transforms.Resize(size=(args.crop_size, args.crop_size)),            
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    model = get_cls_network(args.network, num_classes, use_cuda=True, parallel=True, pretrained=False)      

    saved_state_dict = torch.load(args.model_path)
    model.load_state_dict(saved_state_dict)
    model.eval()
    
    inference_image = composed_transforms(Image.open(os.path.join(args.data_dir, args.image_path)).convert('RGB')).unsqueeze(0).cuda()
    shallow_feature, _ = model(inference_image)
    shallow_feature = shallow_feature.data
    
    interpolate = torch.nn.Upsample(size=(256, 256), mode='bilinear')
    cam_image = interpolate(shallow_feature).sum(dim = 1)[0].cpu().numpy()

    cam_save_path = os.path.join(f"{img_path_prefix}",f"{args.network}_{dataset_name}_{args.image_path.split('/')[-1]}")
    plt.imsave(cam_save_path, cam_image / np.max(cam_image), cmap='viridis')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--network', type=str, default='resnet18', help='alexnet,vgg16,resnet18,resnet50,resnet101,densenet121,densenet201')
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--crop_size', type=int, default=256)

    main(parser.parse_args())
