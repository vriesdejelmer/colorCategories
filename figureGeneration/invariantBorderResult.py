    #loading external libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from scipy.signal import find_peaks
from scipy.ndimage import convolve
from scipy.stats import norm
import os, sys
from pathlib import Path

    #setting cwd and paths
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

from stimulusGen.hue_conversion import HSVColor #RGBHueCircle

#data_folder = '../data/invariantBorders/objectClassLumControl_f40/'
#data_folder = '../data/invariantBorders/rgbCircle_f40/'
#data_folder = '/Users/vriesdejelmer/Dropbox/pythonProjects/categoricalAnalysis/data/invariantBorders/resnet34_f40/'
#data_folder = '../data/invariantBorders/rgbCircleLumControl_f40/'
#data_folder = '../data/invariantBorders/rgbCircle_f40/'
data_folder = '../data/invariantBorders/objectClass_f40/'

border_array_4 = np.load(data_folder + 'border_map_4.npy')
border_array_5 = np.load(data_folder + 'border_map_5.npy')
border_array_6 = np.load(data_folder + 'border_map_6.npy')
border_array_7 = np.load(data_folder + 'border_map_7.npy')
border_array_8 = np.load(data_folder + 'border_map_8.npy')
border_array_9 = np.load(data_folder + 'border_map_9.npy')
border_array = border_array_6+border_array_7+border_array_8+border_array_5+border_array_4+border_array_9

sigma = 5
label_size = 24
caption_size = 24
legend_size = 20
x = np.linspace(norm.ppf(0.05,0,sigma), norm.ppf(0.95,0,sigma), 5)
prob_values = norm.pdf(x,0,sigma)
smoothed = convolve(border_array, prob_values/sum(prob_values), mode='wrap')

(peak_points,_) = find_peaks(border_array, width=2, rel_height=0.55, prominence=15)

plt.figure(figsize=(16,10))

gs = grd.GridSpec(3, 1, height_ratios=[3,1,1], width_ratios=[10], wspace=0.3)

plt.subplot(gs[0])
raw_handle, = plt.plot(np.arange(0,100, 1.0), border_array, linewidth=1.5, color='grey', label='Raw Count')
plt.xlim(0,100)


smoothed_handle, = plt.plot(np.arange(0,100, 1.0), smoothed, linewidth=6, color='skyblue', alpha=0.8, label='Smoothed')
peaks_handle, = plt.plot(peak_points,smoothed[peak_points], 'ro', label='Detected Peaks')
plt.legend(handles=[raw_handle, smoothed_handle, peaks_handle], fontsize=legend_size)
plt.ylabel('Transition count', fontsize=label_size)
plt.xlim(0,100)

plt.subplot(gs[1])
hue_conversion = HSVColor(brightness=1.0) #hue_conversion = RGBHueCircle() use in case of of RGB color space

image_grid = hue_conversion.getHueArray()
plt.imshow(image_grid, interpolation='nearest',aspect='auto')
for border_point in peak_points:
    plt.axvline(x=border_point, color='k', linestyle ="--", linewidth=3)
plt.yticks([])


hues = np.empty(10)
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


plt.subplot(gs[2])
plt.imshow(categorical_array,interpolation='nearest',aspect='auto')

plt.yticks([])
plt.xlabel('Hue (%)', fontsize=18)
plt.text(-8,-66,'A', fontsize=caption_size, fontweight='bold')
plt.text(-8,-12,'B', fontsize=caption_size, fontweight='bold')
plt.text(-8,8,'C', fontsize=caption_size, fontweight='bold')
plt.savefig('../data/figures/circularRGBinvariance.png')