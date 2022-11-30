#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 23:56:34 2022

@author: vriesdejelmer
"""
import os, sys
from pathlib import Path

import matplotlib.pyplot as plt

file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent.parent, 'generalModules')))

from figureFunctions import plotStackFor

names  = ['Object Trained - Full', 'Random Hue Shift - Full', 'Random Weights - Full']
folders = ['objectClass_f40/', 'randomHueShift_f40/', 'resnet18_seeded_f40/']

label_size = 24
caption_size = 24
tick_size = 16
part = 6
plot_index = 1
plt.figure(figsize=(20,7))
plt.subplots_adjust(hspace=0.15, wspace=0.05)
for folder, name in zip(folders, names):

    plt.subplot(1,3,plot_index); plot_index += 1
    
    plotStackFor(part, folder, training_bands=True)
    
    if plot_index % 3 == 2:
        plt.ylabel('Network Iterations', fontsize=label_size)
        
    plt.xlabel('Sample Hue (%)', fontsize=label_size)
    
    plt.xticks([])
    plt.title(name, fontsize=22)


plt.savefig('../../data/figures/appendix/' + 'appendix_figure8.png')