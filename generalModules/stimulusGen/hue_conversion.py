#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:36:19 2020

Manages the hue to rgb process

Includes conversion from
- HSV (colorsys)
- Lab (check cropping beyond [0,1] before usage)
- RGB hue (hue circle in plane R+G+B=1.5 with center at mid-grey)

@author: vriesdejelmer
"""
import numpy as np
import colorsys
from .adapted_color_conv import lab2rgb
import random

class HueColor:

    def toIntRGB(self, hue, brightness=None):
        (r_float, g_float, b_float) = self.toRGB(hue, brightness)
        r_int = int(round(r_float*255))
        g_int = int(round(g_float*255))
        b_int = int(round(b_float*255))
        return (r_int, g_int, b_int)

    def toRGB(self, hue, brightness=None):
        hue = hue % 1.0
        return self._convertHueToRGB(hue, brightness)

        #for visualization purposes
    def getHueArray(self, thickness=20):
        hue_array = np.zeros((100,3))
        for index, hue in enumerate(np.arange(0,1,0.01)):
            hue_array[index, :] = self.toRGB(hue)

        hue_array = np.repeat(np.expand_dims(hue_array,axis=0), repeats=thickness, axis=0)
        return hue_array


class HSVColor(HueColor):

    def __init__(self, brightness=1.0, saturation=1.0):
        self.saturation = saturation
        self.brightness = brightness

    def _convertHueToRGB(self, hue, customBrightness=None):
        if customBrightness is None:
            return colorsys.hsv_to_rgb(hue, self.saturation, self.brightness)
        else:
            return colorsys.hsv_to_rgb(hue, self.saturation, customBrightness)

class HSVColorBrightRange(HueColor):

    def __init__(self, min_brightness=0.9, saturation=1.0):
        self.saturation = saturation
        self.min_brightness = min_brightness

    def _convertHueToRGB(self, hue, customBrightness=None):
        brightness = random.uniform(self.min_brightness, 1.0)
        if customBrightness is None:
            return colorsys.hsv_to_rgb(hue, self.saturation, brightness)
        else:
            return colorsys.hsv_to_rgb(hue, self.saturation, customBrightness)



class LabColor(HueColor):

    hue_steps = 10000

    def __init__(self, brightness=70.0, radius=100):
            #we require some circular coords
        radians = np.linspace(0, np.pi*2, self.hue_steps)
        x_grid = radius * np.sin(radians)
        y_grid = radius * np.cos(radians)

        z_grid = np.ones((self.hue_steps))*brightness
        circular_samples = np.vstack([z_grid, x_grid, y_grid])
        circular_samples = np.expand_dims(circular_samples.transpose(), axis=0)
        self.rgb_array = lab2rgb(circular_samples)


    def _convertHueToRGB(self, hue, brightness=None):
        hue_index = int(hue*self.hue_steps) % self.hue_steps
        return (self.rgb_array[0,hue_index,0], self.rgb_array[0,hue_index,1], self.rgb_array[0,hue_index,2])


class RGBHueCircle(HueColor):

    hue_steps = 10000

    def __init__(self, radius=0.61):    #0.61 is about the biggest "hue" circle we can make
        radians = np.linspace(0, np.pi*2, self.hue_steps)
        x_coords = np.cos(-radians) * radius    #neg radians because we want to go clockwise
        y_coords = [0.0]*self.hue_steps
        z_coords = np.sin(-radians) * radius

        self.rgb_circle = np.vstack([x_coords, y_coords, z_coords])
        z_r = np.pi/4
        z_matrix = [[1, 0, 0], [0, np.cos(z_r), -np.sin(z_r)], [0, np.sin(z_r), np.cos(z_r)]]

        x_r = -(np.pi/2-np.arctan(1/np.sqrt(0.5)))
        x_matrix = [[np.cos(x_r), -np.sin(x_r), 0], [np.sin(x_r), np.cos(x_r),0], [0,0,1]]

        rot_matrix = np.matmul(z_matrix, x_matrix)

        self.rgb_circle = np.dot(rot_matrix, self.rgb_circle).transpose() + 0.5

    def _convertHueToRGB(self, hue, brightness=None):
        hue_index = int(hue*self.hue_steps) % self.hue_steps
        return (self.rgb_circle[hue_index,0], self.rgb_circle[hue_index,1], self.rgb_circle[hue_index,2])
