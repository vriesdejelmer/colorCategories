#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
import matplotlib.gridspec as grd
from crossCorrelation import runCrossCorrelation

#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

from stimulusGen.hue_conversion import HSVColor
from figureFunctions import plotBorderCountFor, getReciprocalCategories

fig = plt.figure(figsize=(40,20))
gs0 = grd.GridSpec(1, 2, width_ratios=[5,2], figure=fig)
gs00 = grd.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0], height_ratios=[4,1,1], width_ratios=[10], wspace=0.3)

data_folder = 'objectClass_f40'

sigma = 5
label_size = 30
caption_size = 36
legend_size = 24
tick_size = 20
hue_conversion = HSVColor(brightness=1.0)

    #subplot A
plt.subplot(gs00[0])
peak_points, border_array = plotBorderCountFor(data_folder, 350, legend_on=True) 
plt.xlim(0,100)
plt.text(-8,300,'A', fontsize=caption_size, fontweight='bold')
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

    #subplot B
plt.subplot(gs00[1])
image_grid = hue_conversion.getHueArray()
plt.imshow(image_grid, interpolation='nearest',aspect='auto')
for border_point in peak_points:
    plt.axvline(x=border_point, color='k', linestyle ="--", linewidth=3)
plt.yticks([])
plt.xticks(fontsize=tick_size)
plt.text(-8,10,'B', fontsize=caption_size, fontweight='bold')


    #subplot C
categorical_array = getReciprocalCategories(peak_points, border_array, hue_conversion)
plt.subplot(gs00[2])
plt.imshow(categorical_array,interpolation='nearest',aspect='auto')

    #subplot D
plt.xticks(fontsize=tick_size)
plt.yticks([])
plt.xlabel('Sample Hue (%)', fontsize=label_size)
plt.text(-8,8,'C', fontsize=caption_size, fontweight='bold')

plt.subplot(gs0[1])
runCrossCorrelation()

plt.savefig('../data/figures/manuscript/' + 'figure2.png')