#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer
"""

import os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import colorsys
from scipy import stats


#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

def getCentralHueMatrix(parts, model_count):
    return np.array([[(x/parts+y/(parts*model_count)) for x in range(parts)] for y in range(model_count)])

    #output folder
model_count = 150
part = 7

label_size = 24
title_size = 28
titles = ['Object Trained', 'Natural vs Artificial']    
full_folder = ['objectClass_f40/', 'nat_v_manmade/']

plt.figure(figsize=(20,7))
plt.subplots_adjust(hspace=0.02, wspace=0.1)

for column in range(len((titles))):

    data_folder = '/Users/vriesdejelmer/Dropbox/pythonProjects/categoricalAnalysis/data/invariantBorders/'
    
    
    folder = data_folder + full_folder[column]

    plt.subplot(1,len(titles),column+1)

    focal_hues_matrix = getCentralHueMatrix(part, model_count)
    
    image_grid = [[colorsys.hsv_to_rgb(x, 1.0, 1.0) for x in np.arange(0,1,0.01)]]*20
    
    stacked_map = np.zeros((model_count, 100, 3))
    
    for model_index in range(model_count):
        color_map = np.load(folder + 'color_map_' + str(part) + '_' + str(model_index) + '.npy')
        mode = stats.mode(color_map, axis=1).mode
        stacked_map[model_index, :, :] = mode[:,0,:]
    
    stacked_map = np.hstack((stacked_map, np.expand_dims(stacked_map[:,-1,:], axis=1)))
    
    plt.imshow(stacked_map,interpolation='nearest',aspect='auto')
    plt.yticks([])
    plt.xlim(0, 100)
    
    plt.xlabel('Hue (%)', fontsize=label_size)
            
    for model_index in range(model_count):
        for hue in focal_hues_matrix[model_index]:
            hueX = hue * 100
            plt.plot(hueX, model_index, 'k|')

    plt.title(titles[column], fontsize=title_size)    
    
plt.savefig('../data/figures/manuscript/' + 'figure3.png')