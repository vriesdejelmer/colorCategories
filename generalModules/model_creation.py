import torch
from torch import nn, flatten
from torchvision import models
import torch.optim as optim
from vqvae import model as vae_model
from vqvae.class_model import ClassModel

def getIdentityModel(model_type, pretrained=True):

    model_list = loadPredefinedModels(model_type, 1, pretrained)
    model_list = replaceOutputLayer(model_type + '_identity', model_list, None)

    return model_list


def getModels(model_type, count, output_num, pretrained=True, frozen=True):

    model_list = loadPredefinedModels(model_type, count, pretrained)

    if frozen:
        model_list = freezeParameters(model_list)

    model_list = replaceOutputLayer(model_type, model_list, output_num)

    return model_list

def freezeParameters(model_list):
    for frozen_model in model_list:
        for param in frozen_model.parameters():
            param.requires_grad = False

    return model_list


def replaceOutputLayer(model_type, model_list, output_num):

    if model_type.startswith('resnet'):
        for model in model_list:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, output_num)
    elif model_type.startswith('vqvae'):
        for model_index, model in enumerate(model_list):
            model_list[model_index] = ClassModel(model, output_num)
    elif model_type == 'alexnet' or model_type == 'vggnet':
        for model in model_list:
            model.classifier[6] = nn.Linear(4096, output_num)
    elif model_type == 'resnet18_identity' or model_type == 'resnet50_identity':
        for model in model_list:
            num_ftrs = model.fc.in_features
            model.fc = nn.Identity()

    return model_list


def loadPredefinedModels(model_type, count, pretrained):

    if model_type == 'resnet18':
        modelList = [models.resnet18(pretrained=pretrained) for _ in range(count)]
    elif model_type == 'resnet18_0':
        modelList = [getPartialResnet(pretrained, 0) for _ in range(count)]
    elif model_type == 'resnet18_1':
        modelList = [getPartialResnet(pretrained, 1) for _ in range(count)]
    elif model_type == 'resnet18_2':
        modelList = [getPartialResnet(pretrained, 2) for _ in range(count)]
    elif model_type == 'resnet18_3':
        modelList = [getPartialResnet(pretrained, 3) for _ in range(count)]
    elif model_type == 'resnet18_4':
        modelList = [getPartialResnet(pretrained, 4) for _ in range(count)]

    elif model_type == 'resnet18_shifted':
        modelList = [hueShiftedResnet() for _ in range(count)]
    elif model_type == 'resnet18_mono':
        modelList = [monochromatResnet() for _ in range(count)]
    elif model_type == 'resnet18_random_color':
        modelList = [randomColorResnet() for _ in range(count)]
    elif model_type == 'resnet18_categorical':
        modelList = [categoricalColorResnet() for _ in range(count)]
    elif model_type == 'resnet18_lab':
        modelList = [labTrainedResnet() for _ in range(count)]
    elif model_type == 'resnet18_overlap_color':
        modelList = [overlapColorResnet() for _ in range(count)]
    elif model_type == 'resnet18_many_color':
        modelList = [disjunctColorWordsResnet() for _ in range(count)]
    elif model_type == 'resnet34':
        modelList = [models.resnet34(pretrained=pretrained) for _ in range(count)]
    elif model_type == 'resnet50':
        modelList = [models.resnet50(pretrained=pretrained) for _ in range(count)]
    elif model_type == 'alexnet':
        modelList = [models.alexnet(pretrained=pretrained) for _ in range(count)]
    elif model_type == 'vggnet':
        modelList = [models.vgg19(pretrained=pretrained) for _ in range(count)]
    elif model_type == 'vqvae_resnet18':
        modelList = [vqvaeModel() for _ in range(count)]

    return modelList


def saveModels(model_optim_list, location, numerical_id, model_name='model'):
    for (model_index, trained_model, optimizer) in model_optim_list:
        model_dict = {
            "model_state": trained_model.state_dict(),
            "optim_state": optimizer.state_dict()
        }
        torch.save(model_dict, location + model_name + '_' + str(numerical_id) + '_' + str(model_index) + '.pth')


def loadSavedModels(model_indices, data_folder, classes, index, model_name='model', learning_rate=0.001, train=False):
    model_optim_list = []
    for model_index in model_indices:
        resnet_model = models.resnet18()
        [resnet_model] = replaceOutputLayer('resnet18', [resnet_model], classes)
        model_dict = torch.load(data_folder + model_name + '_' + str(index) + '_' + str(model_index) + '.pth')
        resnet_model.load_state_dict(model_dict['model_state'])

        optimizer = optim.SGD(resnet_model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(model_dict['optim_state'])

        if train:
            resnet_model.train()
        else:
            resnet_model.eval()
        model_optim_list.append((model_index, resnet_model, optimizer))
    return model_optim_list


def randomColorResnet():

    resnet_model = models.resnet18()
    [resnet_model] = replaceOutputLayer('resnet18', [resnet_model], 14)
    model_dict = torch.load('../models/wordColorModels/random_color_14words.pth', map_location='cpu')
    resnet_model.load_state_dict(model_dict['model_state'])

    return resnet_model


def overlapColorResnet():

    resnet_model = models.resnet18()
    [resnet_model] = replaceOutputLayer('resnet18', [resnet_model], 33)
    model_dict = torch.load('../models/wordColorModels/color_overlap_33words_0_0.pth', map_location='cpu')
    resnet_model.load_state_dict(model_dict['model_state'])

    return resnet_model


def disjunctColorWordsResnet():

    resnet_model = models.resnet18()
    [resnet_model] = replaceOutputLayer('resnet18', [resnet_model], 60)
    model_dict = torch.load('../models/wordColorModels/color_60words_0_0.pth', map_location='cpu')
    resnet_model.load_state_dict(model_dict['model_state'])

    return resnet_model


def categoricalColorResnet():

    resnet_model = models.resnet18()
    [resnet_model] = replaceOutputLayer('resnet18', [resnet_model], 7)
    model_dict = torch.load('../data/networks/categorical_0_0.pth', map_location='cpu')
    resnet_model.load_state_dict(model_dict['model_state'])

    return resnet_model


def hueShiftedResnet():
    model = models.resnet18(pretrained=False)
    data = torch.load('../models/cnnVariations/hueShifted/model_best.pth.tar', map_location='cpu')
    state_dict = data['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()

        #don't remember what this is for.....
    print('Should figure out what this does')
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


def labTrainedResnet():

    model = models.resnet18()
    data = torch.load('../data/networks/resnet18_lab_trichromat_original.pth.tar', map_location='cpu')
    model.load_state_dict(data['state_dict'])

    return model


def monochromatResnet():

    for x in range(100):
        print('THIS LOADS THE HUE_SHIFTED NETWORK, FIX BEFORE USING')
    return None
    model = models.resnet18()
    data = torch.load('../models/cnnVariations/hueShifted/checkpoint.pth.tar', map_location='cpu')
    model.load_state_dict(data['state_dict'])

    return model

def vqvaeModel():

    weights_path='../models/resnet18_vqvave_backbone/model_14.pth'
    weights=torch.load(weights_path, map_location='cpu')

    hidden = 128
    k = 128
    kl = 128
    backbone = { 'arch_name': 'resnet18',
                'layer_name': 'area4' }

    model = vae_model.Backbone_VQ_VAE(hidden, k=k, kl=kl, num_channels=3, colour_space='rgb2rgb', task=None,
                                      out_chns=3, cos_distance=False, use_decor_loss=False, backbone=backbone)
    model.load_state_dict(weights)

    return model.backbone_encoder

def getPartialResnet(pretrained, area_index):
    base_model = models.resnet18(pretrained=pretrained)
    partial_model = PartialResnet(base_model, area_index)
    return partial_model


class PartialResnet(nn.Module):
    def __init__(self, model, area_index):
        super(PartialResnet, self).__init__()

        self.model = self.freezeParameters(model)
        self.base = nn.Sequential(*list(model.children())[:area_index+4])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if area_index == 0 or area_index == 1:
            self.fc = nn.Linear(200704, 1000)
        elif area_index == 2:
            self.fc = nn.Linear(100352, 1000)
        elif area_index == 3:
            self.fc = nn.Linear(50176, 1000)
        elif area_index == 4:
            self.fc = nn.Linear(512, 1000)

    def freezeParameters(self, model):
        for param in model.parameters():
            param.requires_grad = False

        return model


    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

