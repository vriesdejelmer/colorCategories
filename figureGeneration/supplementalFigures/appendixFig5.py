#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

import os, sys
from pathlib import Path

#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

from stimulusGen.hue_conversion import RGBHueCircle
from figureFunctions import plotBorderCountFor, getReciprocalCategories

hue_conversion = RGBHueCircle()
data_folders = ['rgbCircle_f40/', 'rgbCircleLumControl_f40/']
label_size = 24
caption_size = 24
tick_size = 20
legend_size = 20
letters = ['A','B','C','D','E','F']

plt.figure(figsize=(25,10))

for type_index, data_folder in enumerate(data_folders):
    
    gs = grd.GridSpec(3, 2, height_ratios=[3,1,1], wspace=0.15)
    
    plt.subplot(gs[type_index])
    peak_points, border_array = plotBorderCountFor(data_folder, y_limit=200)
    plt.xticks([])
    plt.yticks(fontsize=tick_size)
    plt.ylabel('Transition Count', fontsize=label_size)
    
        #border on spectrum
    plt.subplot(gs[type_index+2])
    image_grid = hue_conversion.getHueArray()
    plt.imshow(image_grid, interpolation='nearest',aspect='auto')
    for border_point in peak_points:
        plt.axvline(x=border_point, color='k', linestyle ="--", linewidth=3)
    plt.yticks([])
    plt.xticks([])
        
    
        #reciprocal prototypes
    plt.subplot(gs[type_index+4])
    categorical_array = getReciprocalCategories(peak_points, border_array, hue_conversion)
    plt.imshow(categorical_array,interpolation='nearest',aspect='auto')
    plt.xticks(fontsize=tick_size)
    plt.yticks([])
    start_index = type_index*3
    plt.xlabel('Sample Hue (%)', fontsize=18)
    plt.text(-13,-69,letters[start_index], fontsize=caption_size, fontweight='bold')
    plt.text(-8,-12,letters[start_index+1], fontsize=caption_size, fontweight='bold')
    plt.text(-8,8,letters[start_index+2], fontsize=caption_size, fontweight='bold')

plt.savefig('../../data/figures/appendix/' + 'appendix_figure5.png')
