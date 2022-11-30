#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer
"""

import os, sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import matplotlib.gridspec as grd
from matplotlib.patches import Rectangle
import colorsys
from scipy import stats

#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

    #import local functions
from datasets.hue_word_datasets import FocalColorWordDataset
from stimulusGen.word_stimulus_generation import SingleWordStimulusGenerator
from stimulusGen.hue_conversion import HSVColor
from figureFunctions import getCentralHueMatrix, plotStackFor

parts = 6
hue_range = 0.2

caption_size = 48
label_size = 42

model_count = 150
thickness = 20
model_num = 12

class SampleProperties:

    def __init__(self):
        self.words = ['color', 'color', 'color', 'color', 'color', 'color', 'color']
        self.focal_hues = [x/parts for x in range(parts)]
        self.hue_range = 0.2/parts
        self.samples_per_class = {'train': 1}

data_transform = transforms.Compose([transforms.ToTensor()])
stim_gen = SingleWordStimulusGenerator(transform=data_transform, color_convertor=HSVColor(), font_size=50)

data_props = SampleProperties()
image_dataset = FocalColorWordDataset('train', data_props, 0, stimulus_generator=stim_gen)
data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=6, shuffle=False, num_workers=0)
folder = 'objectClass_f40'
data_folder = f"/Users/vriesdejelmer/Dropbox/pythonProjects/categoricalAnalysis/data/invariantBorders/{folder}/"


fig = plt.figure(figsize=(40,30))
gs0 = grd.GridSpec(2, 1, height_ratios=[1,4], figure=fig)
gs00 = grd.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs0[0], width_ratios=[1,1,1,1,1,1])

    #Subplot A
for images, labels in data_loader:
    for index, image in enumerate(images):
        ax1 = fig.add_subplot(gs00[index]) #gs00[first_row, index%2]
        plt.imshow(image.numpy().transpose(1,2,0))
        plt.xticks([])
        plt.yticks([])
    break



focal_hues_matrix = getCentralHueMatrix(parts, model_count)
image_grid = [[colorsys.hsv_to_rgb(x, 1.0, 1.0) for x in np.arange(0,1,0.01)]]*thickness


gs01 = gs0[1].subgridspec(4, 1, height_ratios=[1,2,1,4])

    #Subplot B
width = 100/parts*0.2
ax2 = fig.add_subplot(gs01[0, 0])
plt.imshow(image_grid,interpolation='nearest',aspect='auto')
plt.yticks([])
currentAxis = plt.gca()
for hue in focal_hues_matrix[model_num]:
    center_x = hue * 100
    currentAxis.add_patch(Rectangle((center_x - width/2, -1), width, 41, facecolor='Black', fill=True, alpha=0.2))
    currentAxis.add_patch(Rectangle((center_x - width/2, -1), width, 41, fill=False))


    #Color map for C and D
color_map = np.load(data_folder + 'color_map_' + str(parts) + '_' + str(model_num) + '.npy')


    #Subplot C
ax3 = fig.add_subplot(gs01[1, 0])
plt.imshow(color_map.transpose(1,0,2),interpolation='nearest',aspect='auto')
plt.yticks([])
plt.ylabel('Random\n samples', fontsize=label_size)

    #Subplot D
ax4 = fig.add_subplot(gs01[2, 0])
transition_signal = stats.mode(color_map, axis=1).mode
plt.imshow(transition_signal.transpose(1,0,2),interpolation='nearest',aspect='auto')
plt.yticks([])
ax4.yaxis.set_label_coords(-0.01,0.5)
plt.ylabel('Mode', fontsize=label_size)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["left"].set_visible(False)

    #Figure E
ax5 = fig.add_subplot(gs01[3, 0])
plotStackFor(parts, folder, training_bands=True)
plt.yticks([])
plt.xlim(0, 100)
plt.ylabel('Network Iterations', fontsize=label_size)
plt.xlabel('Sample Hue (%)', fontsize=label_size)

    #Add Lettering
plt.text(-8,-320,'A', fontsize=caption_size, fontweight='bold')
plt.text(-8,-180,'B', fontsize=caption_size, fontweight='bold')
plt.text(-8,-130,'C', fontsize=caption_size, fontweight='bold')
plt.text(-8,-40,'D', fontsize=caption_size, fontweight='bold')
plt.text(-8,10,'E', fontsize=caption_size, fontweight='bold')


plt.savefig('../data/figures/manuscript/' + 'figure1.png')