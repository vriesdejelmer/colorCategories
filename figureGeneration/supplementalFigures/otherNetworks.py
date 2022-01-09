#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from scipy.signal import argrelextrema, find_peaks, peak_prominences
from scipy.ndimage import convolve
from scipy.stats import norm

import os, sys
import colorsys
from pathlib import Path

#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

from stimulusGen.hue_conversion import LabColor, HSVColor, RGBHueCircle

network_types = ['objectClass_f40', 'resnet34_f40', 'resnet50_f40', 'resnet101_f40']
names = ['A: Resnet 18', 'B: Resnet 34', 'C: Resnet 50', 'D: Resnet 101']

plt.figure(figsize=(15,20))
sigma = 5
label_size = 24
caption_size = 24
legend_size = 20
tick_size = 16

for plot_index, network_type in enumerate(network_types):
    
    data_folder = '/Users/vriesdejelmer/Dropbox/pythonProjects/categoricalAnalysis/data/invariantBorders/' + network_type + '/'
    
    border_array_4 = np.load(data_folder + 'border_map_4.npy')
    border_array_5 = np.load(data_folder + 'border_map_5.npy')
    border_array_6 = np.load(data_folder + 'border_map_6.npy')
    border_array_7 = np.load(data_folder + 'border_map_7.npy')
    border_array_8 = np.load(data_folder + 'border_map_8.npy')
    border_array_9 = np.load(data_folder + 'border_map_9.npy')
    border_array = border_array_6+border_array_7+border_array_8+border_array_5+border_array_4+border_array_9
        
    x = np.linspace(norm.ppf(0.05,0,sigma), norm.ppf(0.95,0,sigma), 5)
    prob_values = norm.pdf(x,0,sigma)
    smoothed = convolve(border_array, prob_values/sum(prob_values), mode='wrap')
    
    (peak_points,_) = find_peaks(border_array, width=2, rel_height=0.55, prominence=15)
    
    plt.subplot(len(network_types),1,plot_index+1)
    
    raw_handle, = plt.plot(np.arange(0,100, 1.0), border_array, linewidth=1.5, color='grey', label='Raw Count')
    plt.xlim(0,100)
    
    
    smoothed_handle, = plt.plot(np.arange(0,100, 1.0), smoothed, linewidth=6, color='skyblue', alpha=0.8, label='Smoothed')
    peaks_handle, = plt.plot(peak_points,smoothed[peak_points], 'ro', label='Detected Peaks')
    
    plt.text(2,350, names[plot_index], fontsize=24)
    
    if plot_index == 3:
        plt.legend(handles=[raw_handle, smoothed_handle, peaks_handle], fontsize=legend_size)
        plt.xlabel('Hue (%)', fontsize=label_size)
    
    
    plt.ylabel('Transition count', fontsize=label_size)
    
    plt.xlim(0,100)
    plt.ylim(0,400)
    plt.xticks(fontsize=tick_size)
    plt.yticks(ticks=(100,200,300,400), fontsize=tick_size)

    