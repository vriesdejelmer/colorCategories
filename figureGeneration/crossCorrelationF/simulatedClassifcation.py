#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer
"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os, sys
import math
import colorsys
from pathlib import Path
from scipy.stats import ks_2samp, anderson_ksamp, chisquare


#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(__file__)
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

from stimulusGen.stimulus_generation import getCentralHueMatrix

def crossCorr2DHor(array_1, array_2):
    
    array_length = len(array_1)

    deq = deque(np.arange(0,array_length))

    cross_corr_list = np.zeros(array_length)
    for list_index in range(0,array_length):
        correlation_sum = 0
        for index, order_index in enumerate(deq):
            x1, y1 = array_1[index]
            x2, y2 = array_2[order_index]
            correlation_sum += x1*x2 + y1*y2
        deq.rotate(1)

        cross_corr_list[list_index] = correlation_sum

    (max_index,) = np.where(cross_corr_list == max(cross_corr_list))
    max_index[max_index > array_length/2] = max_index[max_index > array_length/2] - array_length
    
    if max_index.shape[0] > 1:
        abs_shifts = np.absolute(max_index)
        min_value = abs_shifts.min()
        (min_indices,) = np.where(abs_shifts == min_value)
        return max_index[min_indices[0]]
    else:
        return max_index[0]
   
def create2DHueMap(data_folder, parts, model_count):
    focal_hues_matrix = getCentralHueMatrix(parts, model_count)

    hue_map = np.array([[0]*100]*model_count, dtype=np.float64)
    hue_map_trans = [[(0,0),]*100]*model_count
    for model_index in range(model_count):
        class_matrix = np.load(data_folder + 'class_matrix_' + str(parts) + '_' + str(model_index) + '.npy')
        focal_hues = focal_hues_matrix[model_index]
        hue_matrix = focal_hues[class_matrix]
        hue_array = stats.mode(hue_matrix, axis=1).mode
        hue_map[model_index] = hue_array[:,0]
        hue_map_trans[model_index] = [(math.cos(a),math.sin(a)) for a in hue_array[:,0]*2*np.pi]

    return hue_map, hue_map_trans

def createBaseMap(model_count, parts, steps):

    focal_hues_matrix = getCentralHueMatrix(parts, model_count)
    
    base_map = np.array([[0]*steps]*model_count, dtype=np.float64)
    base_map_trans = [[(0,0),]*steps]*model_count
    
    base_row = np.floor(np.linspace(0,7-1/100,100)).astype('int')
    
    for model_index in range(model_count):

        index = int(round(steps*((model_index/(model_count*parts))-(1/parts)/2)))
        base_array = np.hstack([base_row[-index:], base_row[:-index]])
        hue_values = focal_hues_matrix[model_index,:]
        base_map[model_index, :] = hue_values[base_array]
        base_map_trans[model_index] = [(math.cos(a),math.sin(a)) for a in base_map[model_index]*2*np.pi]
        
    return base_map, base_map_trans

def displayColorMap(color_map, parts, model_count):

    map_shape = len(color_map), len(color_map[0])
    colored_map = np.zeros((map_shape[0], map_shape[1], 3))    
    
    for x_index in range(map_shape[0]):
        for y_index in range(map_shape[1]):
            colored_map[x_index, y_index, :] = colorsys.hsv_to_rgb(color_map[x_index][y_index], 1.0, 1.0)
    #plt.figure()
    plt.imshow(colored_map,interpolation='nearest',aspect='auto')
    
    focal_hues_matrix = getCentralHueMatrix(parts, model_count)
    for model_index in range(model_count):
        for hue in focal_hues_matrix[model_index]:
            hueX = hue * 100
            plt.plot(hueX, model_index, 'k|', linewidth=0.15)
    plt.yticks([])
    plt.xlim(-0.5, 99.5)
    
    

def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)


display_maps=True 

datasets = ['objectClass_f40', 'categoricalTrained_f40']
names = ['Object Trained', 'Categorically Trained']
parts = 7
steps = 100
model_count = 150

base_map_hue, base_map = createBaseMap(model_count, parts, steps)

if display_maps:
    plt.figure(figsize=(30,8))  
    plt.subplot(1,3,1)
    displayColorMap(base_map_hue, parts, model_count)
    plt.ylabel('Network Iterations', fontsize=24)
    plt.xlabel('Hue (%)', fontsize=24)
    plt.title('Shifting Simulation', fontsize=28)

print('Base is done')

for d_index, dataset in enumerate(datasets):
    data_folder = '../data/invariantBorders/' + dataset + '/'

    hue_map_hue, hue_map = create2DHueMap(data_folder, parts, model_count)
    
    if display_maps:
        plt.subplot(1,3,d_index+2)
        displayColorMap(hue_map_hue, parts, model_count)
        plt.xlabel('Hue (%)', fontsize=24)
        plt.title(names[d_index], fontsize=28)
