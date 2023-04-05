from models.networks import *
import torchvision.models as models
import torch.nn as nn

def restore_from_file(model, restore_from):
    saved_state_dict = torch.load(restore_from)
    new_params = model.state_dict().copy()
    for i, j in zip(saved_state_dict, new_params):
        if (i[0] != 'f') & (i[0] != 's') & (i[0] != 'u'):
            new_params[j] = saved_state_dict[i]
    model.load_state_dict(new_params)

def get_seg_network(network_name, num_classes, restore_from, restore=True, use_cuda=True, parallel=True):
    if network_name == 'fcn32s':
        model = fcn32s(n_classes=num_classes)
        if restore:
            restore_from_file(model, restore_from)
    elif network_name == 'fcn16s':
        model = fcn16s(n_classes=num_classes)
        if restore:
            restore_from_file(model, restore_from)
    elif network_name == 'fcn8s':
        model = fcn8s(n_classes=num_classes)
        if restore:
            restore_from_file(model, restore_from)
    elif network_name == 'deeplabv2':
        model = deeplab(num_classes=num_classes)
        if restore:
            restore_from_file(model, restore_from)
    elif network_name == 'deeplabv3_plus':
        model = deeplabv3_plus(n_classes=num_classes)
    elif network_name == 'segnet':
        model = segnet(n_classes=num_classes)
        if restore:
            vgg16 = models.vgg16(pretrained=True)
            model.init_vgg16_params(vgg16)
    elif network_name == 'icnet':
        model = icnet(n_classes=num_classes)
    elif network_name == 'contextnet':
        model = contextnet(n_classes=num_classes)
    elif network_name == 'sqnet':
        model = sqnet(n_classes=num_classes)
    elif network_name == 'pspnet':
        model = pspnet(n_classes=num_classes)
    elif network_name == 'unet':
        model = unet(n_classes=num_classes)
    elif network_name == 'linknet':
        model = linknet(n_classes=num_classes)
    elif network_name == 'frrna':
        model = frrn(n_classes=num_classes, model_type='A')
    elif network_name == 'frrnb':
        model = frrn(n_classes=num_classes, model_type='B')

    if parallel:
        model = nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()

    return model
