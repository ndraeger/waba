import torch.nn as nn
import models.networks as models

def get_cls_network(network_name, num_classes, use_cuda=True, parallel=True, pretrained=True):
    if network_name == 'alexnet':
        model = models.alexnet(pretrained=pretrained)
        model.classifier._modules['6'] = nn.Linear(4096, num_classes) 
    elif network_name == 'vgg11':
        model = models.vgg11(pretrained=pretrained)  
        model.classifier._modules['6'] = nn.Linear(4096, num_classes)
    elif network_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)  
        model.classifier._modules['6'] = nn.Linear(4096, num_classes)
    elif network_name == 'vgg19':
        model = models.vgg19(pretrained=pretrained)  
        model.classifier._modules['6'] = nn.Linear(4096, num_classes)
    elif network_name == 'inception':
        model = models.inception_v3(pretrained=pretrained, aux_logits=False)  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif network_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif network_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif network_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif network_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=pretrained)  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif network_name == 'resnext101_32x8d':
        model = models.resnext101_32x8d(pretrained=pretrained)  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif network_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)  
        model.classifier = nn.Linear(1024, num_classes)
    elif network_name == 'densenet169':
        model = models.densenet169(pretrained=pretrained)  
        model.classifier = nn.Linear(1664, num_classes)
    elif network_name == 'densenet201':
        model = models.densenet201(pretrained=pretrained)  
        model.classifier = nn.Linear(1920, num_classes)
    elif network_name == 'regnet_x_400mf':
        model = models.regnet_x_400mf(pretrained=pretrained)  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif network_name == 'regnet_x_8gf':
        model = models.regnet_x_8gf(pretrained=pretrained)  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif network_name == 'regnet_x_16gf':
        model = models.regnet_x_16gf(pretrained=pretrained)  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    if parallel:
        model = nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()

    return model