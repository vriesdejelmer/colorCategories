#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer

"""

import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.signal import find_peaks
from scipy.ndimage import convolve
from scipy.stats import norm
import imageio
from colorsys import hsv_to_rgb

pp_count = 10
targetSteps = 35
nodeSteps = 35
sigma = 0.5
tick_size = 16
caption_fontsize = 20
label_size = 20
title_size = 22
data_folder = "../data/humanData/expData/"
human_folder = "../data/humanData/"

def getRespMatrices(resp_dict):

    targetSelections = np.ones((resp_dict['nodeSteps'],resp_dict['targetSteps']))*-1
    responseTimes = np.ones((resp_dict['nodeSteps'],resp_dict['targetSteps']))*-1

    for (target, offset, resp) in zip(resp_dict['targetOffsets'], resp_dict['nodeOffsets'], resp_dict['trialResponses']):
        targetSelections[offset, target] = (resp*resp_dict['nodeSteps'])+offset

    for (target, offset, respTime) in zip(resp_dict['targetOffsets'], resp_dict['nodeOffsets'], resp_dict['trialTimes']):
        responseTimes[offset, target] = respTime

    targetSelections /= (7*resp_dict['nodeSteps'])
    
    return targetSelections, responseTimes

def getTransitionData(allTarSel):
    transition_hist = np.zeros((35), dtype=int)
    transitions_all = np.zeros((10,35), dtype=int)

    for index in range(pp_count):
        color_map = allTarSel[:,:,index]
        left_skip = np.column_stack([color_map[:,-1], color_map[:,:-1]])
        transition_map = left_skip-color_map
        transition_map[transition_map != 0] = 1
        transition_hist += transition_map.astype(int).sum(axis=0)
        transitions_all[index, :] = transition_map.astype(int).sum(axis=0)
    
    x = np.linspace(norm.ppf(0.05,0,sigma), norm.ppf(0.95,0,sigma), 3)
    prob_values = norm.pdf(x,0,sigma)
    smoothed = convolve(transition_hist, prob_values/sum(prob_values), mode='wrap')
    (peak_points,_) = find_peaks(transition_hist, width=0.33, rel_height=0.4, prominence=5, distance=3)
    
    return transition_hist, smoothed, peak_points, transitions_all

def getCategoricalColors(border_array, peak_points):
    
    steps = len(border_array)
    height = 10
    hues = np.empty(len(peak_points))
    categorical_array = np.empty((height,steps,3))

    for cat_index in range(len(peak_points)):
        left_border = peak_points[cat_index]
        right_index = (cat_index+1) % len(peak_points)
        right_border = peak_points[right_index]
        full_sum = sum(border_array[left_border:right_border])

        if right_index == 0:
            full_sum = sum(border_array[left_border:]) + sum(border_array[:right_border])
            right_border += steps

        hue_sum = 0
        for array_index in range(left_border, right_border):
            if right_border >= (steps-1):
                hue_sum += (border_array[array_index-(steps-1)]/full_sum) * (array_index-(steps-0.5))
            else:
                hue_sum += (border_array[array_index]/full_sum) * (array_index+0.5)

        hues[cat_index] = hue_sum

        (r,g,b) = hsv_to_rgb((hue_sum/steps)%1, 1.0, 1.0)
        categorical_array[:,left_border:min(right_border,steps),0] = r
        categorical_array[:,left_border:min(right_border,steps),1] = g
        categorical_array[:,left_border:min(right_border,steps),2] = b
        
        if right_index == 0:
            categorical_array[:,:(right_border-steps),0] = r
            categorical_array[:,:(right_border-steps),1] = g
            categorical_array[:,:(right_border-steps),2] = b
            
    return categorical_array

allTarSel = np.zeros((35,35, pp_count))
allRespTimes = np.zeros((35,35, pp_count))
allAge = np.zeros(pp_count)
allSex = []

for index in range(pp_count):

    data_file = open(f"{data_folder}P{index}.json")
    data_dict = json.load(data_file)
    
    targetSelection, responseTimes = getRespMatrices(data_dict)
    
    allTarSel[:,:,index] = targetSelection
    allRespTimes[:,:,index] = responseTimes
    
allTarSelForMean = allTarSel.copy()
leftSlice = allTarSelForMean[:,:7,:]
leftSlice[leftSlice > 0.75] = 1 - leftSlice[leftSlice > 0.75]
rightSlice = allTarSelForMean[:,-7:,:]
rightSlice[rightSlice < 0.25] = rightSlice[rightSlice < 0.25] + 1
allTarSelForMean[:,:7,:] = leftSlice
allTarSelForMean[:,-7:,:] = rightSlice

meanTarSel = np.mean(allTarSelForMean, axis=2)
medianTarSel = np.median(allTarSel, axis=2)
meanRespTimes = np.mean(allRespTimes, axis=2)

fig = plt.figure(figsize=(20,20))

for index in range(pp_count):
    plt.subplot(3,4,index+1)
    plt.imshow(allTarSel[:,:,index], cmap="hsv")
    for model_index in range(nodeSteps):
        hueX = model_index/7
        for ticks in range(7):
            plt.plot(hueX+ticks*(targetSteps/7)-0.5, model_index, 'k|')

transition_hist, smoothed, peak_points, transitions_all = getTransitionData(allTarSel)
categorical_array = getCategoricalColors(transition_hist, peak_points)

fig = plt.figure(figsize=(20,10))

outer = gridspec.GridSpec(1, 2, width_ratios=[1,1])
ax1 = plt.subplot(outer[0])
stim = imageio.imread(f"{human_folder}stim_example.png")
plt.imshow(stim)
plt.yticks([])
plt.xticks([])
plt.text(30,90,'A', fontsize=caption_fontsize, fontweight='bold', color="white")
plt.title('Stimulus Example', fontsize=title_size)

inner = gridspec.GridSpecFromSubplotSpec(3, 1,
                    subplot_spec=outer[1],height_ratios=[5,2,1])
              

ax1 = plt.subplot(inner[0])
plt.imshow(meanTarSel, extent=(0, 35, 35, 0), cmap="hsv", aspect='auto')
for model_index in range(nodeSteps):
    hueX = model_index/7
    for ticks in range(7):
        plt.plot(hueX+ticks*(targetSteps/7), model_index, 'k|')
    for peak in peak_points:
        plt.axvline(x=peak, color='w', linestyle ="--", linewidth=3)

plt.xticks([])
plt.yticks(np.linspace(0, 35, 6), fontsize=tick_size)
plt.ylabel('Hue Shifts Choices', fontsize=label_size)
plt.title('Selection and Choice Hue', fontsize=title_size)
plt.text(-4,3,'B', fontsize=caption_fontsize, fontweight='bold')


ax2 = plt.subplot(inner[1])

plt.plot(transition_hist, linewidth=1, color='grey')
plt.plot(smoothed, linewidth=4, color='skyblue')
for peak in peak_points:
    plt.plot(peak, smoothed[peak], 'ro', markersize=8, mfc="None")
    

plt.xlim([0,34])
plt.xticks([])
plt.title('Transition Count', fontsize=title_size)
plt.yticks(np.linspace(0, 200, 3), fontsize=tick_size)
plt.ylabel('Count', fontsize=label_size)
plt.text(-4.2,240,'C', fontsize=caption_fontsize, fontweight='bold')

ax3 = plt.subplot(inner[2])
plt.imshow(categorical_array, extent=(0, 15, 35, 0), cmap="hsv", aspect='auto')
label_repl = [f"{i:.0f}" for i in np.linspace(0,100,6)]
plt.xticks(ticks = [0, 3, 6, 9, 12, 15], labels=label_repl, fontsize=tick_size)
plt.yticks([])
plt.title('Category Prototypes', fontsize=title_size)
plt.xlabel('Hue (%)', fontsize=label_size)

plt.text(-1.88,5,'D', fontsize=caption_fontsize, fontweight='bold')

plt.savefig('../data/figures/manuscript/' + 'figure4.png')