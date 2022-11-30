#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from torchvision import transforms
from torch.utils.data import DataLoader
import os, sys
from pathlib import Path

#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir).parent))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

from stimulusGen.hue_conversion import HSVColor
from figureFunctions import plotBorderCountFor, getReciprocalCategories
from stimulusGen.word_stimulus_generation import SingleWordSavedBackgroundStimulusGenerator
from datasets.hue_word_datasets import FocalColorWordDataset

hue_conversion = HSVColor()
data_folder = 'objectClassLumControl_f40/'
label_size = 24
caption_size = 24
tick_size = 20
legend_size = 20
parts = 6
letters = ['A','B','C','D','E','F']


class SampleProperties:

    def __init__(self):
        self.words = ['color', 'color', 'color', 'color', 'color', 'color', 'color']
        self.focal_hues = [x/parts for x in range(parts)]
        self.hue_range = 0.2/parts
        self.samples_per_class = {'train': 1}

data_transform = transforms.Compose([transforms.ToTensor()])
stim_gen = SingleWordSavedBackgroundStimulusGenerator(transform=data_transform, color_convertor=HSVColor(), font_size=40)

data_props = SampleProperties()
image_dataset = FocalColorWordDataset('train', data_props, 0, stimulus_generator=stim_gen)
data_loader = DataLoader(image_dataset, batch_size=6, shuffle=False, num_workers=0)

fig = plt.figure(figsize=(30,15))
gs0 = grd.GridSpec(1, 2, width_ratios=[1,2], figure=fig)
gs00 = grd.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs0[0], width_ratios=[1,1], height_ratios=[1,1,1])

    #Subplot A
for images, labels in data_loader:
    for index, image in enumerate(images):
        ax1 = fig.add_subplot(gs00[index]) #gs00[first_row, index%2]
        plt.imshow(image.numpy().transpose(1,2,0))
        plt.xticks([])
        plt.yticks([])
    break


gs01 = grd.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1], height_ratios=[3,1,1])

plt.subplot(gs01[0])
peak_points, border_array = plotBorderCountFor(data_folder, y_limit=200)
plt.xticks([])
plt.yticks(fontsize=tick_size)
plt.ylabel('Transition Count', fontsize=label_size)

    #border on spectrum
plt.subplot(gs01[1])
image_grid = hue_conversion.getHueArray()
plt.imshow(image_grid, interpolation='nearest',aspect='auto')
for border_point in peak_points:
    plt.axvline(x=border_point, color='k', linestyle ="--", linewidth=3)
plt.yticks([])
plt.xticks([])
    

    #reciprocal prototypes
plt.subplot(gs01[2])
categorical_array = getReciprocalCategories(peak_points, border_array, hue_conversion)
plt.imshow(categorical_array,interpolation='nearest',aspect='auto')
plt.xticks(fontsize=tick_size)
plt.yticks([])

plt.xlabel('Sample Hue (%)', fontsize=18)
plt.text(-70,-69,'A', fontsize=caption_size, fontweight='bold')
plt.text(-13,-69,'B', fontsize=caption_size, fontweight='bold')
plt.text(-8,-12,'C', fontsize=caption_size, fontweight='bold')
plt.text(-8,8,'D', fontsize=caption_size, fontweight='bold')

plt.savefig('../data/figures/appendix/' + 'appendix_figure3.png')
