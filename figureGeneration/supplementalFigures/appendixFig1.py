#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer
"""

import matplotlib.pyplot as plt

import os, sys
from pathlib import Path


#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent.parent, 'generalModules')))

from figureFunctions import plotBorderCountFor

folder_names = ['objectClass_f40', 'resnet18_alt_inst_f40', 'resnet34_f40', 'resnet50_f40', 'resnet101_f40']
names = ['A: Resnet 18 (ori)', 'B: Resnet 18 (rep)', 'C: Resnet 34', 'D: Resnet 50', 'E: Resnet 101']
y_limit = 400

sigma = 5
label_size = 24
caption_size = 24
legend_size = 20
tick_size = 16
title_size = 36
rgb_on = False
legend_on = False

plt.figure(figsize=(15,20))
for plot_index, folder_name in enumerate(folder_names):
   
    plt.subplot(len(folder_names),1,plot_index+1)
   
    if plot_index == len(folder_names)-1:
        legend_on = True

    plotBorderCountFor(folder_name, y_limit, legend_on) 
    
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    
    plt.text(2,y_limit*0.85, names[plot_index], fontsize=24)
    
    if plot_index == 0:
        plt.title("Transition Count", fontsize=title_size)
    
    if plot_index == (len(folder_names)-1):
        plt.xlabel('Sample Hue (%)', fontsize=label_size)
    else:
        plt.xticks([])
    
    
    plt.ylabel('Count', fontsize=label_size)
    
    plt.xticks(fontsize=tick_size)
    plt.yticks(ticks=range(100,y_limit+1,100), fontsize=tick_size)
    
plt.savefig('../../data/figures/appendix/' + 'appendix_figure1.png')
    