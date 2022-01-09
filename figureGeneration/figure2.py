    #loading external libraries
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
import matplotlib.gridspec as grd
from scipy.signal import find_peaks
from scipy.ndimage import convolve
from scipy.stats import norm
from crossCorrelation import runCrossCorrelation

    #setting cwd and paths
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

    #import local functions
from stimulusGen.hue_conversion import HSVColor

fig = plt.figure(figsize=(40,20))
gs0 = grd.GridSpec(1, 2, width_ratios=[5,2], figure=fig)
gs00 = grd.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0], height_ratios=[4,1,1], width_ratios=[10], wspace=0.3)
data_folder = '../data/invariantBorders/objectClass_f40/'

border_array_4 = np.load(data_folder + 'border_map_4.npy')
border_array_5 = np.load(data_folder + 'border_map_5.npy')
border_array_6 = np.load(data_folder + 'border_map_6.npy')
border_array_7 = np.load(data_folder + 'border_map_7.npy')
border_array_8 = np.load(data_folder + 'border_map_8.npy')
border_array_9 = np.load(data_folder + 'border_map_9.npy')
border_array = border_array_4+border_array_5+border_array_6+border_array_7+border_array_8+border_array_9

sigma = 5
label_size = 30
caption_size = 36
legend_size = 24
tick_size = 20
x = np.linspace(norm.ppf(0.05,0,sigma), norm.ppf(0.95,0,sigma), 5)
prob_values = norm.pdf(x,0,sigma)
smoothed = convolve(border_array, prob_values/sum(prob_values), mode='wrap')

(peak_points,_) = find_peaks(border_array, width=2)

plt.subplot(gs00[0])
raw_handle, = plt.plot(np.arange(0,100, 1.0), border_array, linewidth=1.5, color='grey', label='Raw Count')
plt.xlim(0,100)
plt.text(-8,300,'A', fontsize=caption_size, fontweight='bold')

smoothed_handle, = plt.plot(np.arange(0,100, 1.0), smoothed, linewidth=6, color='skyblue', alpha=0.8, label='Smoothed')
peaks_handle, = plt.plot(peak_points,smoothed[peak_points], 'ro', label='Detected Peaks')
plt.legend(handles=[raw_handle, smoothed_handle, peaks_handle], fontsize=legend_size)
plt.ylabel('Transition count', fontsize=label_size)
plt.xlim(0,100)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.subplot(gs00[1])
hue_conversion = HSVColor(brightness=1.0)
image_grid = hue_conversion.getHueArray()
plt.imshow(image_grid, interpolation='nearest',aspect='auto')
for border_point in peak_points:
    plt.axvline(x=border_point, color='k', linestyle ="--", linewidth=3)
plt.yticks([])
plt.xticks(fontsize=tick_size)
plt.text(-8,10,'B', fontsize=caption_size, fontweight='bold')

hues = np.empty(8)
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

    hues[cat_index] = hue_sum

    (r,g,b) = hue_conversion._convertHueToRGB((hue_sum/100)%1)
    categorical_array[:,left_border:right_border,0] = r
    categorical_array[:,left_border:right_border,1] = g
    categorical_array[:,left_border:right_border,2] = b

    if right_index == 0:
        categorical_array[:,:right_border-99,0] = r
        categorical_array[:,:right_border-99,1] = g
        categorical_array[:,:right_border-99,2] = b


plt.subplot(gs00[2])
plt.imshow(categorical_array,interpolation='nearest',aspect='auto')

plt.xticks(fontsize=tick_size)
plt.yticks([])
plt.xlabel('Hue (%)', fontsize=label_size)
plt.text(-8,8,'C', fontsize=caption_size, fontweight='bold')

plt.subplot(gs0[1])

runCrossCorrelation()
