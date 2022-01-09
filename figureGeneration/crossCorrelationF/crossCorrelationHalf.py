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
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent.parent, 'generalModules')))

from correlationFunctions import create2DBaseMap, create2DHueMap, calculateShiftsSingleComp, displayColorMap, crossCorr2DHor
from stimulusGen.stimulus_generation import getCentralHueMatrix

display_maps=True 

datasets = ['objectClass_f40', 'categoricalTrained_f40']
names = ['Object Trained', 'Categorically Trained']
parts = 7
steps = 100
model_count = 150

focal_hues_matrix = getCentralHueMatrix(parts, model_count)
base_map, base_map_2D = create2DBaseMap(model_count, parts, steps, focal_hues_matrix)

if display_maps:
    plt.figure(figsize=(10,12))  
    plt.subplot(1,3,1)
    displayColorMap(base_map)
    plt.title('Shifting')

base_shifts = calculateShiftsSingleComp(base_map_2D, crossCorr2DHor, model_count)

print('Base is done')

for d_index, dataset in enumerate(datasets):
    data_folder = '../../data/invariantBorders/' + dataset + '/'

    hue_map, hue_map_2D = create2DHueMap(data_folder, parts, model_count, steps, focal_hues_matrix)
        
    if display_maps:
        plt.subplot(1,3,d_index+2)
        displayColorMap(hue_map)
        plt.title(names[d_index])

    if d_index == 0: 
        object_shifts = calculateShiftsSingleComp(hue_map_2D, crossCorr2DHor, model_count)
    else:
        categorical_shifts = calculateShiftsSingleComp(hue_map_2D, crossCorr2DHor, model_count)  

plt.figure(figsize=(10,12))     

bin_ranges = np.arange(-0.5, 9.5)
(values_orig, bins_orig) = np.histogram(object_shifts, bin_ranges)
(values_cat, bins_cat) = np.histogram(categorical_shifts, bin_ranges)                   
(values_shift, bins_shift) = np.histogram(base_shifts, bin_ranges)

bin_values = np.arange(0, 9)

plt.bar(bin_values, values_orig/sum(values_orig), width=0.8, color="deepskyblue", ec="grey")
plt.plot(bin_values, values_cat/sum(values_cat), linewidth=4, color='green')
plt.plot(bin_values, values_shift/sum(values_shift), linewidth=4, color='orchid')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim((0, 8))
plt.xlabel('Optimal Shift', fontsize=24)
plt.ylabel('Relative Frequency', fontsize=24)
plt.legend(['Categorically Trained', 'Diagonal Baseline', 'Current'], fontsize=18)

###### stats
norm_cat = values_cat/sum(values_cat)
norm_shift = values_shift/sum(values_shift)
norm_orig = values_orig/sum(values_orig)
    
distr = np.vstack((norm_orig, norm_shift))
print(np.sum(distr.min(axis=0)))

distr = np.vstack((norm_orig, norm_cat))
print(np.sum(distr.min(axis=0)))

print('Original')
print(repr(values_orig))
print('Categorical')
print(repr(values_cat))
print('Continuous')
print(repr(values_shift))