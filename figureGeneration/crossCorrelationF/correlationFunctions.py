#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 23:20:54 2021

@author: vriesdejelmer
"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import statistics
import colorsys
from astropy.stats import circcorrcoef

def crossCorrCirc(array_1, array_2):
    
    array_length = array_1.shape[0]
    
    array_1_rad = array_1.copy()*np.pi
    array_2_rad = array_2.copy()*np.pi
    
    cross_corr_list = np.zeros(array_length)    #a place to store the cross correlations
    
    for offset in range(0,array_length):
        shifted_array_2 = np.hstack([array_2_rad[offset:], array_2_rad[:(offset)]])
        cross_corr_list[offset] = circcorrcoef(array_1_rad, shifted_array_2)
        
    max_index = getClosestMaxIndex(cross_corr_list)
    return max_index

def crossCorr2DHor(array_1, array_2):
    
    array_length = array_1.shape[0]
    
    cross_corr_list = np.zeros(array_length)    #a place to store the cross correlations
    
    for offset in range(0,array_length):
        
        shifted_array_x = np.hstack([array_2[offset:, 0], array_2[:(offset), 0]])
        shifted_array_y = np.hstack([array_2[offset:, 1], array_2[:(offset), 1]])
      
        cross_corr_list[offset] = np.dot(array_1[:,0], shifted_array_x) + np.dot(array_1[:,1], shifted_array_y)
        
    return getClosestMaxIndex(cross_corr_list)


def getClosestMaxIndex(cross_corr_list):

    array_length = len(cross_corr_list)
    (max_index,) = np.where(cross_corr_list == max(cross_corr_list))
    
        #second half the array is easier the other way around
    max_index[max_index > array_length/2] = max_index[max_index > array_length/2] - array_length
    
    if max_index.shape[0] > 1:
        abs_shifts = np.absolute(max_index)
        min_value = abs_shifts.min()
        (min_indices,) = np.where(abs_shifts == min_value)
        return max_index[min_indices[0]]
    else:
        return max_index[0]
       

def create2DHueMap(data_folder, parts, model_count, steps, focal_hues_matrix):
    
    hue_map = np.array([[0]*steps]*model_count, dtype=np.float64)
    hue_map_2D = np.zeros((model_count, steps,2))
    
    for model_index in range(model_count):
        class_matrix = np.load(data_folder + 'class_matrix_' + str(parts) + '_' + str(model_index) + '.npy')
        focal_hues = focal_hues_matrix[model_index]
        hue_matrix = focal_hues[class_matrix]
        hue_array = stats.mode(hue_matrix, axis=1).mode
        hue_map[model_index] = hue_array[:,0]
        hue_map_2D[model_index,:,0] = [math.cos(a) for a in hue_array[:,0]*2*np.pi]
        hue_map_2D[model_index,:,1] = [math.sin(a) for a in hue_array[:,0]*2*np.pi]

    return hue_map, hue_map_2D

def create2DBaseMap(model_count, parts, steps, focal_hues_matrix):
    
    base_map = np.array([[0]*steps]*model_count, dtype=np.float64)
    base_map_2D = np.zeros((model_count, steps, 2))
    
    base_row = np.floor(np.linspace(0,7-1/100,100)).astype('int')
    
    for model_index in range(model_count):

        index = int(round(steps*((model_index/(model_count*parts))-(1/parts)/2)))
        base_array = np.hstack([base_row[-index:], base_row[:-index]])
        hue_values = focal_hues_matrix[model_index,:]
        base_map[model_index, :] = hue_values[base_array]
        base_map_2D[model_index,:,0] = [math.cos(a) for a in base_map[model_index]*2*np.pi]
        base_map_2D[model_index,:,1] = [math.sin(a) for a in base_map[model_index]*2*np.pi]
        
    return base_map, base_map_2D

def calculateShiftsAllRows(hue_map, corr_function, model_count):

    shifts = np.zeros(model_count * (model_count))
    counter = 0
    for current_row in range(model_count):
        for row in range(model_count):
            shifts[counter] = corr_function(hue_map[row], hue_map[current_row])
            
            counter += 1

    if counter != shifts.size:
        print("THERE SEEMS TO BE A BIG PROBLEM!!!!!!")
    return shifts


def calculateShiftsOtherRows(hue_map, corr_function, model_count):
    
    shifts = np.zeros(model_count * (model_count-1))
    counter = 0
    for current_row in range(model_count):
        all_rows = np.arange(model_count)
        row_selection = all_rows != current_row
        
        for row in all_rows[row_selection]: #range(model_count):
            shifts[counter] = corr_function(hue_map[row], hue_map[current_row])
            counter += 1

    if counter != shifts.size:
        print("THERE SEEMS TO BE A BIG PROBLEM!!!!!!")
    return shifts

def calculateShiftsSingleComp(hue_map, corr_function, model_count):

    shifts = np.zeros(int((model_count) * (model_count-1)/2))
    counter = 0
    for current_row in range(model_count):
        
        for row in range(current_row+1, model_count):
            shifts[counter] = abs(corr_function(hue_map[row], hue_map[current_row]))
            counter += 1

    if counter != shifts.size:
        print("THERE SEEMS TO BE A BIG PROBLEM!!!!!!")

    return shifts


def displayColorMap(color_map):

    map_shape = len(color_map), len(color_map[0])
    colored_map = np.zeros((map_shape[0], map_shape[1], 3))    
    
    for x_index in range(map_shape[0]):
        for y_index in range(map_shape[1]):
            colored_map[x_index, y_index, :] = colorsys.hsv_to_rgb(color_map[x_index][y_index], 1.0, 1.0)
    plt.figure()
    plt.imshow(colored_map)



def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)
