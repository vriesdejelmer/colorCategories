    #loading external libraries
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import os, sys
import matplotlib.gridspec as grd
from pathlib import Path

    #setting cwd and paths
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

    #import local functions
from stimulusGen.hue_conversion import HSVColor


    #local variables
spectrum_height = 20
parts = 7
data_folder = 'collapse/'
repetition_count = 12 #number of times we ran the evolutionary algorithm
top_count = 10
original_borders = [3, 10, 23, 37, 56, 78, 88]


    #collecting data
combined_borders = np.zeros((repetition_count*top_count, 7), dtype=np.float32)
print(combined_borders.shape)
border_frequency = np.zeros((100), dtype=np.int16)
image_grid = [[colorsys.hsv_to_rgb(x, 1.0, 1.0) for x in np.arange(0,1,0.01)]]*spectrum_height
for it_index in range(1, repetition_count+1):
    print(it_index)
    genetic_borders = np.load('../data/genetics/' + data_folder + str(it_index)  + '/genetic_border_matrix_' + str(parts) + 'parts.npy')
    combined_borders[(it_index-1)*top_count:it_index*top_count] = genetic_borders[0:top_count,:]

    for borders_index in range(top_count):
        for border in np.array(genetic_borders[borders_index, :]):
            borderX = border * 100
            border_frequency[int(np.round(borderX)%100)] += 1

median_borders = np.median(combined_borders, axis=0)
standard_dev_borders = np.std(combined_borders, axis=0)


##### plotting

plt.savefig('../data/figures/genetic_results.png')

plt.figure(figsize=(15,6))
gs = grd.GridSpec(2, 1, height_ratios=[3,1], width_ratios=[10], wspace=0.3)
plt.subplot(gs[0])
plt.plot(border_frequency/(repetition_count*top_count), linewidth=6, color='skyblue')

for border_point in original_borders:
    plt.axvline(x=border_point, color=(0.3,0.3,0.3), linestyle ="--", linewidth=3)

median_borders = np.median(combined_borders, axis=0)
sign = 1
for border, sd_border in zip(median_borders, standard_dev_borders):
    plt.errorbar(border*100, 0.2+0.05*sign, xerr=sd_border*100, linewidth=3, fmt='o', capsize=10, capthick=3, ecolor='k')
    sign = -sign

plt.ylabel('Border Incidence (prop)', fontsize=18)
plt.xlim(0,100)

plt.subplot(gs[1])
hue_conversion = HSVColor(brightness=1.0)
image_grid = hue_conversion.getHueArray()
plt.imshow(image_grid, interpolation='nearest',aspect='auto')
for border_point in original_borders:
    plt.axvline(x=border_point, color='k', linestyle ="--", linewidth=3)
plt.yticks([])
plt.xlim(-0.5,99.5)
plt.xlabel('Hue (%)', fontsize=18)
