#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer
"""

import os, sys
from pathlib import Path

import matplotlib.pyplot as plt
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent.parent, 'generalModules')))

from figureFunctions import plotStackFor, plotBorderCountFor

names  = ['Resnet 18', 'Alexnet', 'GoogLeNet', 'VGG Net 19', 'MobileNet V2', 'DenseNet']
folder_names = ['objectClass_f40/', 'alexnet_f40/', 'googlenet_f40/', 'vggnet_f40/', 'mobilenetv2_f40/', 'densenet_f40/']

y_limit = 375
label_size = 24
caption_size = 24
title_size = 32
legend_size = 20
tick_size = 16
part = 7
legend_on = False

plt.figure(figsize=(25,30))

plt.subplots_adjust(wspace=0.1)
plot_index = 1
for folder, name in zip(folder_names, names):
    plt.subplot(6,2,plot_index) 
    
    if plot_index == 1:
        plt.title("Transition Count", fontsize=title_size)

    if plot_index == 11:
        legend_on = True
    
    plotBorderCountFor(folder, y_limit, legend_on=legend_on, legend_position='upper right') 
    
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    name_index = int((plot_index-1)/2)
    plt.text(15,y_limit*0.8, names[name_index], fontsize=24)
    
    if plot_index == 11:
        plt.xlabel('Sample Hue (%)', fontsize=label_size)
    else: 
        plt.xticks([])
    
    plt.ylabel('Count', fontsize=label_size)
    plot_index += 2
    
    
plot_index = 2
for folder, name in zip(folder_names, names):

    if plot_index == 2:
        plt.title("Classification", fontsize=title_size)    

    plt.subplot(6,2,plot_index); plot_index += 2
    
    plotStackFor(part, folder, training_bands=True)
    
    plt.ylabel('Network It.', fontsize=label_size)
    
    if plot_index == 12:
        plt.xlabel('Sample Hue (%)', fontsize=label_size)
    else: 
        plt.xticks([])
        
plt.savefig('../../data/figures/appendix/' + 'appendix_figure9.png')
