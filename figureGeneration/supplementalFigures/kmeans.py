    #load externals
import numpy as np
import colorsys
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.gridspec as grd
import sys, os
from pathlib import Path


    #set paths and cwd correctly
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))


    #load local code
from stimulusGen.hue_conversion import HSVColor
from general_functions import cart2pol, prop2unit


    #variables
frequency_matrix = np.load('../../data/ImagenetPixData/rgb_frequency.npy')
steps = 101
brightness_steps = 128
borders = [3, 10, 23, 37, 56, 78, 88]
categories = len(borders)


    #calculate some basic frequency for the HSV color space
def getFrequencyArrays(freq_matrix, min_bright=0.2, sum_threshold=1.0, bright_threshold=0.99, sat_threshold=0.99):
    
    hue_freq = np.zeros(steps, dtype=int)
    brightness_freq = np.zeros(brightness_steps, dtype=int)
    sat_freq = np.zeros(steps, dtype=int)
    hue_maxB_freq = np.zeros(steps, dtype=int)

    for red in range(256):
        for green in range(256):
            for blue in range(256):
                (hue, sat, brightness) = colorsys.rgb_to_hsv(red/255, green/255, blue/255)
                brightness_freq[int(brightness*(brightness_steps-1))] += freq_matrix[red, green, blue]

                if brightness > min_bright:
                    sat_freq[int(sat*(steps-1))] += freq_matrix[red, green, blue]
                    if sat+brightness > sum_threshold:
                    
                        hue_freq[int(hue*(steps-1))] += freq_matrix[red, green, blue]
                        if brightness > bright_threshold and sat > sat_threshold:
                            hue_maxB_freq[int(hue*(steps-1))] += freq_matrix[red, green, blue]
    
    return (hue_freq, brightness_freq, sat_freq, hue_maxB_freq)

def getHueList(freq_matrix, hue_maxB_freq):
    hue_list = np.zeros(hue_maxB_freq.sum())
    counter = 0
    for red in range(256):
        for green in range(256):
            for blue in range(256):
                (hue, sat, brightness) = colorsys.rgb_to_hsv(red/255, green/255, blue/255)
                if brightness > 0.99 and sat > 0.99:
                    count = freq_matrix[red, green, blue]
                    for _ in range(count):
                        hue_list[counter] = hue
                        counter += 1
                        
    return hue_list

    #collect data
(hue_frequency, brightness_frequency, sat_frequency, hue_maxB_frequency) = getFrequencyArrays(frequency_matrix)

hue_list = getHueList(frequency_matrix, hue_maxB_frequency)
hue_x, hue_y = prop2unit(hue_list)
hue_cartesian = np.vstack([hue_x, hue_y]).transpose()


    #perform kmeans    
clustering = KMeans(n_clusters=categories, random_state=5)
clustering.fit(hue_cartesian)

rhos, phis = cart2pol(clustering.cluster_centers_[:,0], clustering.cluster_centers_[:,1])
centers = (phis % (np.pi*2))/(np.pi*2)


    # create frequency array
frequency_array = np.zeros((categories, steps), dtype=np.int)
for label in range(categories):
    sub_selection = hue_list[clustering.labels_ == label]
    for hue in sub_selection:
        frequency_array[label, int(hue*100)] += 1


    #plot figure
plt.figure(figsize=(15,7))
gs = grd.GridSpec(2, 1, height_ratios=[12,1], width_ratios=[1], hspace=0.05)

plt.subplot(gs[0])

for label in range(categories):
    current_freq = frequency_array[label,:]
    hue_sum = 0
    for step in range(steps):
            #hack to make sure the class crossing from 99 to 0 gets the right color
        if current_freq[0] != 0 and step < 50:
            hue_sum += (step+100)*current_freq[step]
        hue_sum += step*current_freq[step]
    hue_average = hue_sum/current_freq.sum()
    rgb_color = colorsys.hsv_to_rgb(hue_average/100, 1.0, 1.0)
    plt.bar(range(steps), current_freq/current_freq.sum(), color=rgb_color, alpha=0.7)

plt.ylabel('Proportion', fontsize=18)

for border in borders:
    plt.axvline(x=border, color=[0.3, 0.3, 0.3], linestyle ="--", linewidth=3)
    
plt.xlim([-0.5, 99.5])
plt.xticks([])

plt.subplot(gs[1])

hue_conversion = HSVColor(brightness=1.0)
image_grid = hue_conversion.getHueArray()

plt.imshow(image_grid, interpolation='nearest', aspect='auto')
plt.xlabel('Hue (%)', fontsize=18)
plt.yticks([])