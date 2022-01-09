    #loading external libraries
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

    #setting cwd and paths
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

    #import local functions
from datasets.hue_word_datasets import FocalColorWordDataset
from stimulusGen.word_stimulus_generation import SingleWordStimulusGenerator
from stimulusGen.hue_conversion import HSVColor
from stimulusGen.stimulus_generation import getCentralHueMatrix

class DataProperties:

    def __init__(self, words, focal_hues, hue_range, samples_per_class):
        self.words = words
        self.focal_hues = focal_hues
        self.hue_range = hue_range
        self.samples_per_class = samples_per_class


    #output folder
data_folder = '../testing/outputs/'

data_transform = transforms.Compose([transforms.ToTensor()])
stim_gen = SingleWordStimulusGenerator(transform=data_transform, color_convertor=HSVColor(), font_size=50)

words = ['color', 'color', 'color', 'color', 'color', 'color', 'color']
parts = 7
hue_range = 0.01
caption_fontsize = 48
label_size = 42
focal_hues = [x/parts for x in range(parts)]

samples_per_class ={'train': 1}
data_props = DataProperties(words, focal_hues, hue_range, samples_per_class)
image_dataset = FocalColorWordDataset('train', data_props, 0, stimulus_generator=stim_gen)

data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=6, shuffle=False, num_workers=0)

fig = plt.figure(figsize=(40,30))
gs0 = grd.GridSpec(2, 1, height_ratios=[1,4], figure=fig)
gs00 = grd.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs0[0], width_ratios=[1,1,1,1,1,1])

for images, labels in data_loader:
    for index, image in enumerate(images):
        #first_row = int(np.floor(index/2))
        ax1 = fig.add_subplot(gs00[index]) #gs00[first_row, index%2]
        plt.imshow(image.numpy().transpose(1,2,0))
        plt.xticks([])
        plt.yticks([])
    break

thickness = 20
width = 100/parts*0.2
image_grid = [[colorsys.hsv_to_rgb(x, 1.0, 1.0) for x in np.arange(0,1,0.01)]]*thickness

model_num = 12

data_folder = '/Users/vriesdejelmer/Dropbox/pythonProjects/categoricalAnalysis/data/invariantBorders/objectClass_f40/'
color_map = np.load(data_folder + 'color_map_' + str(parts) + '_' + str(model_num) + '.npy')


    #we want a matrix with all hues (each row holds hue for iteration)
model_count = 150

focal_hues_matrix = getCentralHueMatrix(parts, model_count)

image_grid = [[colorsys.hsv_to_rgb(x, 1.0, 1.0) for x in np.arange(0,1,0.01)]]*20

gs01 = gs0[1].subgridspec(4, 1, height_ratios=[1,2,1,4])

ax2 = fig.add_subplot(gs01[0, 0])
plt.imshow(image_grid,interpolation='nearest',aspect='auto')
plt.yticks([])
currentAxis = plt.gca()
for hue in focal_hues_matrix[model_num]:
    someX = hue * 100
    currentAxis.add_patch(Rectangle((someX - width/2, -1), width, 41, facecolor='Black', fill=True, alpha=0.2))
    currentAxis.add_patch(Rectangle((someX - width/2, -1), width, 41, fill=False))

ax3 = fig.add_subplot(gs01[1, 0])
plt.imshow(color_map.transpose(1,0,2),interpolation='nearest',aspect='auto')
plt.yticks([])

plt.ylabel('Random\n samples', fontsize=label_size)
currentAxis = plt.gca()

ax4 = fig.add_subplot(gs01[2, 0])
transition_signal = stats.mode(color_map, axis=1).mode
plt.imshow(transition_signal.transpose(1,0,2),interpolation='nearest',aspect='auto')
plt.yticks([])
ax4.yaxis.set_label_coords(-0.01,0.5)
plt.ylabel('Mode', fontsize=label_size)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["left"].set_visible(False)

ax5 = fig.add_subplot(gs01[3, 0])

stacked_map = np.zeros((model_count, 100, 3))

for model_index in range(model_count):
    color_map = np.load(data_folder + 'color_map_' + str(parts) + '_' + str(model_index) + '.npy')
    mode = stats.mode(color_map, axis=1).mode
    stacked_map[model_index, :, :] = mode[:,0,:]

stacked_map = np.hstack((stacked_map, np.expand_dims(stacked_map[:,-1,:], axis=1)))

plt.imshow(stacked_map,interpolation='nearest',aspect='auto')
plt.yticks([])
plt.xlim(0, 100)

plt.text(-8,-320,'A', fontsize=caption_fontsize, fontweight='bold')
plt.text(-8,-180,'B', fontsize=caption_fontsize, fontweight='bold')
plt.text(-8,-130,'C', fontsize=caption_fontsize, fontweight='bold')
plt.text(-8,-40,'D', fontsize=caption_fontsize, fontweight='bold')
plt.text(-8,10,'E', fontsize=caption_fontsize, fontweight='bold')


plt.ylabel('Network Iterations', fontsize=label_size)
plt.xlabel('Hue (%)', fontsize=24)
currentAxis = plt.gca()


for model_index in range(model_count):
    for hue in focal_hues_matrix[model_index]:
        hueX = hue * 100
        plt.plot(hueX, model_index, 'k|')
