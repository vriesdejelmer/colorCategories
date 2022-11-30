#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 00:44:08 2022

@author: vriesdejelmer
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema, find_peaks
from scipy.ndimage import convolve
from scipy.stats import norm
from scipy import stats
from stimulusGen.hue_conversion import HSVColor

def getCentralHueMatrix(parts, model_count):
    return np.array([[(x/parts+y/(parts*model_count)) for x in range(parts)] for y in range(model_count)])


def plotStackFor(parts, folder, model_count=150, training_bands=False):

    data_folder = '/Users/vriesdejelmer/Dropbox/pythonProjects/categoricalAnalysis/data/invariantBorders/' + folder + '/'
    focal_hues_matrix = getCentralHueMatrix(parts, model_count)

    stacked_map = np.zeros((model_count, 100, 3))
    
    for model_index in range(model_count):
        color_map = np.load(data_folder + 'color_map_' + str(parts) + '_' + str(model_index) + '.npy')
        mode = stats.mode(color_map, axis=1).mode
        stacked_map[model_index, :, :] = mode[:,0,:]
    
    if training_bands:
        for model_index in range(model_count):
            for hue in focal_hues_matrix[model_index]:
                hueX = hue * 100
                plt.plot(hueX, model_index, 'k|')
    
    stacked_map = np.hstack((stacked_map, np.expand_dims(stacked_map[:,-1,:], axis=1)))
    
    plt.imshow(stacked_map,interpolation='nearest',aspect='auto')
    plt.yticks([])
    plt.xlim(0, 100)
    

def plotBorderCountFor(folder, y_limit, legend_on=False, legend_position='best', legend_size=20, sigma=5):
       
    border_array = getBorderArray(folder)    
    
    x = np.linspace(norm.ppf(0.05,0,sigma), norm.ppf(0.95,0,sigma), 5)
    prob_values = norm.pdf(x,0,sigma)
    smoothed = convolve(border_array, prob_values/sum(prob_values), mode='wrap')
    
    (peak_points,_) = find_peaks(border_array, width=1, rel_height=0.4, prominence=15, height=75, distance=5)
    raw_handle, = plt.plot(np.arange(0,100, 1.0), border_array, linewidth=1.5, color='grey', label='Raw Count')
    plt.xlim(0,100)

    smoothed_handle, = plt.plot(np.arange(0,100, 1.0), smoothed, linewidth=6, color='skyblue', alpha=0.8, label='Smoothed')
    peaks_handle, = plt.plot(peak_points,smoothed[peak_points], 'ro', label='Detected Peaks')
    
    if legend_on:
        plt.legend(handles=[raw_handle, smoothed_handle, peaks_handle], fontsize=legend_size, loc=legend_position)
        
    plt.xlim(0,100)
    plt.ylim(0,y_limit)
    
    plt.yticks(ticks=range(100,y_limit+1,100))
    
    return peak_points, border_array

def getBorderArray(folder):
    data_folder = '/Users/vriesdejelmer/Dropbox/pythonProjects/categoricalAnalysis/data/invariantBorders/' + folder + '/'
    border_array_4 = np.load(data_folder + 'border_map_4.npy')
    border_array_5 = np.load(data_folder + 'border_map_5.npy')
    border_array_6 = np.load(data_folder + 'border_map_6.npy')
    border_array_7 = np.load(data_folder + 'border_map_7.npy')
    border_array_8 = np.load(data_folder + 'border_map_8.npy')
    border_array_9 = np.load(data_folder + 'border_map_9.npy')
    return border_array_6+border_array_7+border_array_8+border_array_5+border_array_4+border_array_9


def getReciprocalCategories(peak_points, border_array, hue_conversion=HSVColor(brightness=1.0)):
    categorical_array = np.empty((15,100,3))

    for cat_index in range(len(peak_points)):
        left_border = peak_points[cat_index]
        right_index = (cat_index+1) % len(peak_points)
        right_border = peak_points[(cat_index+1) % len(peak_points)]
        full_sum = sum(border_array[left_border:right_border])
        if right_index == 0:
            full_sum = sum(border_array[left_border:]) + sum(border_array[:right_border])
            right_border += 100

        hue_sum = 0
        for array_index in range(left_border, right_border):
            if right_border >= 99:
                hue_sum += (border_array[array_index-99]/full_sum) * (array_index-98.5)
            else:
                hue_sum += (border_array[array_index]/full_sum) * (array_index+0.5)

        #hues[cat_index] = hue_sum

        (r,g,b) = hue_conversion._convertHueToRGB((hue_sum/100)%1)
        categorical_array[:,left_border:right_border,0] = r
        categorical_array[:,left_border:right_border,1] = g
        categorical_array[:,left_border:right_border,2] = b

        if right_index == 0:
            categorical_array[:,:right_border-99,0] = r
            categorical_array[:,:right_border-99,1] = g
            categorical_array[:,:right_border-99,2] = b
            
    return categorical_array

