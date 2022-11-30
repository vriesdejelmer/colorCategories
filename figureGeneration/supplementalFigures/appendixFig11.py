#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:34:16 2022

@author: vriesdejelmer
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, sys
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats


#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

    #import local functions
from stimulusGen.word_stimulus_generation import SingleWordStimulusGenerator
from stimulusGen.hue_conversion import HSVColor


    #output folder
data_folder = '../testing/outputs/'

data_transform = transforms.Compose([transforms.ToTensor()])
stim_gen = SingleWordStimulusGenerator(transform=data_transform, color_convertor=HSVColor(), font_size=50)

parts = 7
caption_fontsize = 48
label_size = 48
tick_size = 36
focal_hues = [x/parts for x in range(parts)]

posthoc_tests = 9
thickness = 5
categories = [3, 10, 23, 37, 56, 78, 88]

data_folder = '/Users/vriesdejelmer/Dropbox/pythonProjects/categoricalAnalysis/data/invariantBorders/posthocTesting/'
    
fig = plt.figure(figsize=(40,30))

for index in range(posthoc_tests):
    plt.subplot(posthoc_tests,1,index+1)
    
    color_map = np.load(data_folder + 'color_map_' + str(index) + '_0.npy')
    hue_classes = np.load(data_folder + 'hue_bands_' + str(index) + '.npy')
    
    plt.imshow(color_map.transpose(1,0,2),interpolation='nearest',aspect='auto')
    plt.yticks([])
    
    currentAxis = plt.gca()
    height = color_map.shape[1]
    for class_band in hue_classes:
        width = thickness/7
        currentAxis.add_patch(Rectangle((class_band*100-width/2, -1), width, height, facecolor='Black', fill=True, alpha=0.2))
        currentAxis.add_patch(Rectangle((class_band*100-width/2, -1), width, height, fill=False))
            
    for border_point in categories:
        plt.axvline(x=border_point, color='w', linestyle ="--", linewidth=10)
        
    if index == 4:
        plt.ylabel('Shifted training sessions', fontsize=label_size)
    if index == 8:
        plt.xlabel('Hue (%)', fontsize=label_size)
        plt.xticks([10, 30, 50, 70, 90], fontsize=tick_size)
    else: 
        plt.xticks([])

plt.savefig('../../data/figures/appendix/' + 'appendix_figure11.png')

fig = plt.figure(figsize=(40,30))

for index in range(posthoc_tests):
    plt.subplot(posthoc_tests,1,index+1)
    
    color_map = np.load(data_folder + 'color_map_' + str(index) + '_0.npy')
    hue_classes = np.load(data_folder + 'hue_bands_' + str(index) + '.npy')
    
    transition_signal = stats.mode(color_map, axis=1).mode
    plt.imshow(transition_signal.transpose(1,0,2),interpolation='nearest',aspect='auto')
    plt.yticks([])

    currentAxis = plt.gca()
    height = color_map.shape[1]
    
    for class_band in hue_classes:
        width = thickness/7
        currentAxis.add_patch(Rectangle((class_band*100-width/2, -1), width, height, facecolor='Black', fill=True, alpha=0.2))
        currentAxis.add_patch(Rectangle((class_band*100-width/2, -1), width, height, fill=False))
            
    for border_point in categories:
        plt.axvline(x=border_point, color='w', linestyle ="--", linewidth=10)

    if index == 8:
        plt.xlabel('Hue (%)', fontsize=label_size)
    else: 
        plt.xticks([])
