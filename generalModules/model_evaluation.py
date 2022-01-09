#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:11:25 2020

@author: vriesdejelmer
"""

### This script trains a batch of networks on color bands (slightly shifted for each network)

    #import main packages
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
import torch.nn as nn

    #ensure we use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = nn.Softmax(dim=1)

    #we log data
def findHueClassification(models, dataset_manager, data_folder, index=0, range_properties=None):

    full_range_loader = dataset_manager.getFullRangeLoader(range_properties)
    samples = dataset_manager.getSamplesPerColor()

    border_matrix = np.zeros(100, dtype=int)    #matrix keeping track of borders

    for model_index, model in enumerate(models):

        hue_matrix = classesForHues(model, full_range_loader, samples)
        np.save(data_folder + 'class_matrix_' + str(index) + '_' + str(model_index), hue_matrix)

        focal_hues = dataset_manager.getFocalHues(model_index)
        if focal_hues is not None:
            colored_matrix = getColoredMatrix(hue_matrix, dataset_manager.getHueConvertor(), focal_hues)
            np.save(data_folder + 'color_map_' + str(index) + '_' + str(model_index), colored_matrix)

        addToBorderMatrix(border_matrix, hue_matrix)

    np.save(data_folder + 'border_map_' + str(index), border_matrix)


    #we log data
def findHue2DObjectBorders(models, dataset_manager, data_folder, index=0, range_properties=None):

    full_range_loader = dataset_manager.getFullRangeLoader(range_properties)
    samples = dataset_manager.getSamplesPerColor()

    for model_index, model in enumerate(models):

        hue_matrix = predict2DHueMatrix(model, full_range_loader, samples)
        np.save(data_folder + 'class_matrix_' + str(index) + '_' + str(model_index), hue_matrix)


    #we find predictions for each hue
def classesForHues(frozen_model, full_range_loader, samples):
    hueMatrix = np.zeros([100, samples], dtype=int)
    frozen_model.to(device)
    frozen_model.eval()
    with torch.no_grad():
        for image, hue, var in tqdm(full_range_loader):
            image = image.to(device)
            output = frozen_model(image)
            _, predicted = torch.max(output, 1)
            hueMatrix[hue, var] = predicted.cpu().numpy()
    frozen_model.to('cpu')
    return hueMatrix


    #we find predictions for each hue
def predict2DHueMatrix(frozen_model, full_range_loader, samples):
    hueMatrix = np.zeros([50, 50, samples], dtype=int)
    frozen_model.to(device)
    frozen_model.eval()
    with torch.no_grad():
        for images, (hue1, hue2), var in tqdm(full_range_loader):
            images = images.to(device)
            output = frozen_model(images)
            _, predicted = torch.max(output, 1)
            hueMatrix[hue1, hue2, var] = predicted.cpu().numpy()
    frozen_model.to('cpu')
    return hueMatrix

    #we want to create a nice color-coded version of the map
def getColoredMatrix(hue_matrix, hue_convertor, focal_hues):
    colors = np.array([hue_convertor.toRGB(x) for x in focal_hues])
    return colors[hue_matrix]


    #we want to keep track of borders
def addToBorderMatrix(border_matrix, hue_matrix):
    hue_array = stats.mode(hue_matrix, axis=1).mode
    hue_array = hue_array[:,0]
    subtraction_matrix = np.abs(np.hstack((hue_array[1:], hue_array[0])) - hue_array)
    subtraction_matrix[subtraction_matrix > 1] = 1
    border_matrix += subtraction_matrix


    #we find the RGB predictions
def findRGBpredictions(models, dataset_manager, data_folder, index=0):
        #we create a data_loader that goes through the full spectrum

    full_range_loader = dataset_manager.getFullRangeLoader()
    samples = dataset_manager.getSamplesPerColor()

    for model_index, model in enumerate(models):
        class_matrix, prob_matrix = predictRGBMatrix(model, dataset_manager.getSteps(), full_range_loader, samples)
        np.save(data_folder + 'class_matrix_' + str(index) + '_' + str(model_index), class_matrix)
        np.save(data_folder + 'prob_matrix_' + str(index) + '_' + str(model_index), prob_matrix)


def predictRGBMatrix(frozen_model, steps, full_range_loader, samples):
    rgb_class_matrix = np.zeros([steps, steps, steps, samples], dtype=np.int)
    rgb_prob_matrix = np.zeros([steps, steps, steps, samples], dtype=np.float)
    frozen_model.to(device)
    frozen_model.eval()
    with torch.no_grad():
        for image, (r, g, b), var in tqdm(full_range_loader):
            image = image.to(device)
            output = frozen_model(image)
            _, predicted = torch.max(output, 1)
            probabilities = softmax(output)
            rgb_class_matrix[r, g, b, :] = predicted.cpu().numpy()
            rgb_prob_matrix[r, g, b, :] = probabilities[torch.arange(len(predicted)), predicted].cpu().numpy()

    frozen_model.to('cpu')
    return rgb_class_matrix, rgb_prob_matrix
