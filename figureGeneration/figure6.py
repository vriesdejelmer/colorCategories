import os, sys
from pathlib import Path

#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from torchvision import transforms
from stimulusGen.word_stimulus_generation import ClutteredWordStimulusGenerator
from stimulusGen.hue_conversion import HSVColor
import numpy as np
import colorsys
import random
from general_functions import colorline

caption_size = 32

def getRandomHueColor(brightness=0.5):
    random_color = colorsys.hsv_to_rgb(random.uniform(0.0, 1.0), 1.0, brightness)
    return (int(random_color[0]*255), int(random_color[1]*255), int(random_color[2]*255))


def getCategorySelection(original_borders, splits, cat_prop):

    hue_ranges = []
    for index in range(len(original_borders)):
        next_index = (index+1) % len(original_borders)
        left_border = original_borders[index]
        right_border = original_borders[next_index]
        if right_border < left_border:
            right_border += 1

        range_start = left_border + (right_border - left_border) * cat_prop
        hue_range = (right_border-left_border)/splits
        hue_ranges.append((range_start, range_start+hue_range))

    return hue_ranges


def getSDfromProps(prop_array):

    index_array = np.arange(100)
    squared_array = np.power(index_array,2)
    mean_value = (prop_array * index_array).sum()/prop_array.sum()
    variance_array = squared_array - np.power(mean_value, 2)
    variance = np.sqrt((variance_array * prop_array).sum()/prop_array.sum())

    return variance
data_folder = '../data/wordColor/multiBorderResultFinalCats_f40/'

fig = plt.figure(figsize=(15,7))

gs = grd.GridSpec(2, 1, height_ratios=[2,6], figure=fig)
gs0 = grd.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[0], width_ratios=[1,1,1,1,1,1,1])

data_transform = transforms.Compose([transforms.ToTensor()])
stim_gen = ClutteredWordStimulusGenerator(color_convertor=HSVColor(1.0), transform=data_transform)

words = ['color', 'color', 'color', 'color', 'color', 'color', 'color']
hues = [(-0.07, -0.04), (0.06, 0.07), (0.14, 0.17), (0.27, 0.31), (0.44, 0.49), (0.62, 0.68), (0.83, 0.86)]

for index, (word, hue_range) in enumerate(zip(words, hues)):
    hue = random.uniform(hue_range[0], hue_range[1])
    hue4 = random.uniform(0.0, 1.0)
    hue5 = random.uniform(0.0, 1.0)
    image = stim_gen.getStimImage((hue, hue, hue, hue4, hue5), word=word, background_color=getRandomHueColor())
    plt.subplot(gs0[index])
    plt.imshow(image.numpy().transpose(1,2,0))
    plt.xticks([])
    plt.yticks([])


original_borders = np.array([3, 10, 23, 37, 56, 78, 88])/100

iterations = 15
plt.subplot(gs[1])

prop_values = np.zeros(100)
all_props = np.zeros(100)
errors = np.zeros(100)

for x in range(7):

    for location, cat_prop in enumerate(np.arange(0.0,1.0,0.1)):
        prop_matrix = np.empty((iterations, 100))

        hue_ranges = getCategorySelection(original_borders, 10, cat_prop)

        for it in range(iterations):
            class_matrix = np.load(data_folder + 'class_matrix_' + str(it+(location+1)*100) + '_0.npy')
            incorrect_matrix = class_matrix != x
            incorrect_matrix = incorrect_matrix.transpose()
            prop_matrix[it] = incorrect_matrix.sum(axis=0)/100

        prop_array = np.median(prop_matrix, axis=0)
        error = prop_matrix.std(axis=0)

        lower_x = (hue_ranges[x][0]%1.0) * 100
        higher_x = (hue_ranges[x][1]%1.0) * 100

        if higher_x < lower_x:
            higher_x += 100

        full_array = np.arange(0,100)
        x_selection = np.logical_and(full_array >= lower_x, full_array <= higher_x)
        prop_values[x_selection] = prop_array[x_selection]
        errors[x_selection] = error[x_selection]

        ax = plt.gca()
        ax.set_facecolor((0.75, 0.75, 0.75))

        plt.axvline(x=original_borders[x]*100, color='k', linestyle ="--", linewidth=3)

x_values = np.arange(0,100)
value_selection = prop_values != 0
x_values = x_values[value_selection]
prop_values = prop_values[value_selection]
errors = errors[value_selection]

plt.fill_between(x_values, prop_values-errors, prop_values+errors, alpha=0.5, edgecolor='#646464', facecolor='#646464')
colorline(x_values, prop_values, cmap='hsv')

plt.ylim([0.0, 0.5])
plt.xlim([0, 99])
plt.xlabel('Hue (%)', fontsize=18)
plt.ylabel('Median Error Rate', fontsize=18)

plt.text(-8, 0.68,'A', fontsize=caption_size, fontweight='bold')
plt.text(-8, 0.48,'B', fontsize=caption_size, fontweight='bold')

plt.savefig('../data/figures/manuscript/' + 'figure6.png')