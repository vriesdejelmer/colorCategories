    #loading external libraries
import os, sys
from pathlib import Path
import csv
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math
import matplotlib.gridspec as grd
from torchvision import transforms

    #setting cwd and paths
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

    #import local functions
from general_functions import colorline
from stimulusGen.object_stimulus_generation import SingleObjectStimulusGenerator
from stimulusGen.hue_conversion import HSVColor

    #output folder
data_folder = '../testing/outputs/'

    #
data_transform = transforms.Compose([transforms.ToTensor()])
stim_gen = SingleObjectStimulusGenerator(color_convertor=HSVColor(), transform=data_transform, background_color=0.5)

    #
caption_fontsize = 24
fig = plt.figure(figsize=(15,10))
gs0 = grd.GridSpec(2, 1, height_ratios=[1,4], figure=fig)
gs00 = grd.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs0[0], width_ratios=[1,1,1,1,1,1])
gs01 = grd.GridSpecFromSubplotSpec(7, 2, subplot_spec=gs0[1])

    #
object_types = ['couch', 'axe', 'dolphin', 'sailboat', 'carrot', 'apple']
object_hue = [0.3, 0.15, 0.55, 0.8, 0.1, 0.99]
object_index = [1, 2, 2, 3, 3, 3]


for index, object_type in enumerate(object_types):
    image = stim_gen.getStimImage(object_hue[index], model_phase='train', object_type=object_type, index=object_index[index])
    fig.add_subplot(gs00[index])
    image_matrix = image.numpy()
    plt.imshow(image_matrix.transpose(1,2,0))
    plt.title(object_type)
    plt.xticks([])
    plt.yticks([])


def getHueRanges():

    prop_file = '../data/objectColor/twoObjectsPerCatFinal/1/object_list.csv'
    with open(prop_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    ranges = data[1]
    hue_ranges = []
    for hue_range in ranges:
        lower, upper = map(float, hue_range.strip('( )').split(','))
        if lower < 0: lower +=1
        if upper < 0: upper +=1
        hue_ranges.append((lower, upper))

    return hue_ranges

exp_type = 'twoObjectsPerCat'

classes = 14
object_types = ['strawberry', 'apple', 'crab', 'dog', 'school bus', 'cow', 'dolphin', 'mushroom', 'bird', 'submarine', 'angel', 'sweater', 'sailboat', 'duck']
borders = [3, 10, 23, 37, 56, 78, 88]
borders = [(3, 10), (10, 23), (23, 37), (37, 56), (56,78), (78,88), (88,3)]
hue_ranges = getHueRanges()
iterations = 70

for x in range(classes):
    prop_matrix = np.empty((iterations, 100))
    cumsum_matrix = np.empty((iterations, 100))
    border_matrix = np.empty((iterations, 99))


    for it in range(iterations):
        data_folder = '../data/objectColor/twoObjectsPerCatFinal/' + str(it) + '/'
        class_matrix = np.load(data_folder + 'class_matrix_' + str(x) + '_0.npy')
        correct_matrix = class_matrix == x
        correct_matrix = correct_matrix.transpose()
        prop_matrix[it] = correct_matrix.sum(axis=0)/80
        thresholded = np.array(prop_matrix[it] > 0.75, dtype=np.float)
        thresholded = np.abs(thresholded[:-1] - thresholded[1:])
        border_matrix[it] = thresholded

    prop_array = np.median(prop_matrix, axis=0)
    error = prop_matrix.std(axis=0)

    fig.add_subplot(gs01[x])

    plt.fill_between(range(100), prop_array-error, prop_array+error, alpha=0.5, edgecolor='#646464', facecolor='#646464')
    colorline(range(100), prop_array, cmap='hsv')

    ax = plt.gca()
    ax.set_facecolor((0.75, 0.75, 0.75))


    (upper, lower) = borders[math.floor(x/2)]
    plt.axvline(x=upper, color='k', linestyle ="--", linewidth=3)
    plt.axvline(x=lower, color='k', linestyle ="--", linewidth=3)
    (upper, lower) = hue_ranges[x]

    currentAxis = plt.gca()
    face_color = colorsys.hsv_to_rgb((upper+lower)/2, 1.0, 1.0)
    currentAxis.add_patch(Rectangle((lower*100, -1), (upper-lower)*100, 41, facecolor=face_color, fill=True, alpha=0.5))
    currentAxis.add_patch(Rectangle((lower*100, -1), (upper-lower)*100, 41, color=face_color, fill=False))


    plt.xlim([0, 99])
    plt.ylim([0, 1])

    if x == 12 or x == 13:
        plt.xlabel('Hue (%)', fontsize=18)

    if x == 6:
        plt.ylabel('Median Proportion Correct', fontsize=18)

plt.text(-140,11,'A', fontsize=caption_fontsize, fontweight='bold')
plt.text(-140,8,'B', fontsize=caption_fontsize, fontweight='bold')
