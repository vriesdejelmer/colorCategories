#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:48:19 2020

@author: vriesdejelmer
"""

import numpy as np
import random
from .stimulus_generation import StimulusGenerator


class RectStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,50), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var

    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):
        stim_width, stim_height = self.image_size
        rect_width, rect_height = self.rect_size
        rect_height = round(rect_height + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_height)
        rect_width = round(rect_width + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_width)
        image_array = np.ones((stim_width, stim_height, 3), dtype=np.float32) * 0.5
        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width, :] = [color_int/255 for color_int in foreground_color]

        return image_array


class ShapeStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,80), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var

    def getRectStim(self, foreground_color, stim_x_rel, stim_y_rel):
        stim_width, stim_height = self.image_size
        rect_width, rect_height = self.rect_size
        rect_height = round(rect_height + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_height)
        rect_width = round(rect_width + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_width)
        image_array = np.ones((stim_width, stim_height, 3), dtype=np.float32) * 0.5
        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width, :] = [color_int/255 for color_int in foreground_color]

        return image_array

    def getDiskStim(self, foreground_color, stim_x_rel, stim_y_rel):

        red = foreground_color[0]/255
        green = foreground_color[1]/255
        blue = foreground_color[2]/255

        stim_width, stim_height = self.image_size

        rect_width, rect_height = self.rect_size
        rect_height = round(rect_height + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_height)
        rect_width = round(rect_width + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_width)

        x_coords, y_coords = np.meshgrid(np.arange(rect_width),np.arange(rect_height))
        x_coords = x_coords - rect_width/2
        y_coords = y_coords - rect_height/2
        x_coords = x_coords * (rect_height/rect_width)
        distances = np.sqrt(np.power(x_coords,2) + np.power(y_coords,2))
        stimulus = np.ones((rect_height, rect_width))
        stimulus[distances > rect_width/2] = 0

        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)

        image_array = np.ones((self.image_size[0], self.image_size[1],3), dtype=np.float32)*0.5
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += stimulus*(red-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += stimulus*(green-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += stimulus*(blue-0.5)

        return image_array

    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        if random.random() > 0.5:
            image_array = self.getDiskStim(foreground_color, stim_x_rel, stim_y_rel)
        else:
            image_array = self.getRectStim(foreground_color, stim_x_rel, stim_y_rel)

        return image_array

class DiskStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,80), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var

    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        red = foreground_color[0]/255
        green = foreground_color[1]/255
        blue = foreground_color[2]/255

        stim_width, stim_height = self.image_size
        rect_width, rect_height = self.rect_size

        x_coords, y_coords = np.meshgrid(np.arange(self.rect_size[0]),np.arange(self.rect_size[1]))
        x_coords = x_coords - self.rect_size[0]/2
        y_coords = y_coords - self.rect_size[1]/2
        distances = np.sqrt(np.power(x_coords,2) + np.power(y_coords,2))
        stimulus = np.ones(self.rect_size)
        stimulus[distances > rect_width/2] = 0

        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)

        image_array = np.ones((self.image_size[0], self.image_size[1],3), dtype=np.float32)*0.5
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += stimulus*(red-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += stimulus*(green-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += stimulus*(blue-0.5)

        return image_array

class RingStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,80), ring_thickness=0.1, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.ring_thickness = ring_thickness

    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        red = foreground_color[0]/255
        green = foreground_color[1]/255
        blue = foreground_color[2]/255

        stim_width, stim_height = self.image_size
        rect_width, rect_height = self.rect_size

        x_coords, y_coords = np.meshgrid(np.arange(self.rect_size[0]),np.arange(self.rect_size[1]))
        x_coords = x_coords - self.rect_size[0]/2
        y_coords = y_coords - self.rect_size[1]/2
        distances = np.sqrt(np.power(x_coords,2) + np.power(y_coords,2))
        stimulus = np.ones(self.rect_size)
        stimulus[distances > rect_width/2] = 0.0
        stimulus[distances < (rect_width/2)*(1-self.ring_thickness)] = 0.0

        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)

        image_array = np.ones((self.image_size[0], self.image_size[1],3), dtype=np.float32)*0.5
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += stimulus*(red-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += stimulus*(green-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += stimulus*(blue-0.5)

        return image_array

class GaussStimulusGenerator(StimulusGenerator):


    def multivariate_gaussian(self, size):
        """Return the multivariate Gaussian distribution on array pos."""

        sigma_x = size[0]/17.5
        sigma_y = size[1]/17.5

        x = np.linspace(-20, 20, size[1])
        y = np.linspace(-20, 20, size[0])

        x, y = np.meshgrid(x, y)
        z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
             + y**2/(2*sigma_y**2))))

        return z

class BlobStimulusGenerator(GaussStimulusGenerator):

    def __init__(self, rect_size=(80,80), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var

    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        red = foreground_color[0]/255
        green = foreground_color[1]/255
        blue = foreground_color[2]/255

        stim_width, stim_height = self.image_size
        rect_width, rect_height = self.rect_size
        rect_width = int(rect_width * 1.5)
        rect_height = int(rect_height * 1.5)

        blob = self.multivariate_gaussian((rect_width, rect_height))
        blob = blob/blob.max()

        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)

        image_array = np.ones((self.image_size[0], self.image_size[1],3), dtype=np.float32)*0.5
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += blob*(red-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += blob*(green-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += blob*(blue-0.5)

        return image_array


class InvertedGaussStimulusGenerator(GaussStimulusGenerator):

    def __init__(self, rect_size=(80,80), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var

    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        red = foreground_color[0]/255
        green = foreground_color[1]/255
        blue = foreground_color[2]/255

        stim_width, stim_height = self.image_size
        rect_width, rect_height = self.rect_size
        rect_width = int(rect_width * 1.5)
        rect_height = int(rect_height * 1.5)

        blob = self.multivariate_gaussian((rect_width, rect_height))
        maximum = blob.max()
        blob = blob/(maximum)

        square_blob_dip = np.ones((rect_width, rect_height))
        square_blob_dip -= blob

        #square_blob_dip *= blob.sum()/square_blob_dip.sum()


        x_coords, y_coords = np.meshgrid(np.arange(rect_width),np.arange(rect_height))
        x_coords = x_coords - rect_width/2
        y_coords = y_coords - rect_height/2
        distances = np.sqrt(np.power(x_coords,2) + np.power(y_coords,2))

        square_blob_dip[distances > (self.rect_size[0]/2-1)] = 0.0;

        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)

        image_array = np.ones((self.image_size[0], self.image_size[1],3), dtype=np.float32)*0.5

        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += square_blob_dip*(red-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += square_blob_dip*(green-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += square_blob_dip*(blue-0.5)

        return image_array


class MixedDisk3StimulusGenerator(GaussStimulusGenerator):

    def __init__(self, rect_size=(80,80), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var

    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        red = foreground_color[0]/255
        green = foreground_color[1]/255
        blue = foreground_color[2]/255

        stim_width, stim_height = self.image_size
        rect_width, rect_height = self.rect_size
        rect_width = int(rect_width * 1.5)
        rect_height = int(rect_height * 1.5)

        blob = self.multivariate_gaussian((rect_width, rect_height))
        maximum = blob.max()
        blob = blob/(maximum)

        square_blob_dip = np.ones((rect_width, rect_height))
        square_blob_dip -= blob

        #square_blob_dip *= blob.sum()/square_blob_dip.sum()

        x_coords, y_coords = np.meshgrid(np.arange(rect_width),np.arange(rect_height))
        x_coords = x_coords - rect_width/2
        y_coords = y_coords - rect_height/2
        distances = np.sqrt(np.power(x_coords,2) + np.power(y_coords,2))

        square_blob_dip[distances > (self.rect_size[0]/2-1)] = 0.0;

        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)

        image_array = np.ones((self.image_size[0], self.image_size[1],3), dtype=np.float32)*0.5
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += blob*(0.5-red)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += blob*(0.5-green)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += blob*(0.5-blue)

        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += square_blob_dip*(red-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += square_blob_dip*(green-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += square_blob_dip*(blue-0.5)

        return image_array


class MixedDisk2StimulusGenerator(GaussStimulusGenerator):

    def __init__(self, rect_size=(80,80), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var

    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        red = foreground_color[0]/255
        green = foreground_color[1]/255
        blue = foreground_color[2]/255

        stim_width, stim_height = self.image_size
        rect_width, rect_height = self.rect_size
        rect_width = int(rect_width * 1.5)
        rect_height = int(rect_height * 1.5)

        blob = self.multivariate_gaussian((rect_width, rect_height))
        maximum = blob.max()
        blob = blob/(maximum)

        square_blob_dip = np.ones((rect_width, rect_height))
        square_blob_dip -= blob

        #square_blob_dip *= blob.sum()/square_blob_dip.sum()

        x_coords, y_coords = np.meshgrid(np.arange(rect_width),np.arange(rect_height))
        x_coords = x_coords - rect_width/2
        y_coords = y_coords - rect_height/2
        distances = np.sqrt(np.power(x_coords,2) + np.power(y_coords,2))

        square_blob_dip[distances > (self.rect_size[0]/2-1)] = 0.0;

        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)

        image_array = np.ones((self.image_size[0], self.image_size[1],3), dtype=np.float32)*0.5
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += blob*(red-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += blob*(green-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += blob*(blue-0.5)

        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += square_blob_dip*(0.5-red)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += square_blob_dip*(0.5-green)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += square_blob_dip*(0.5-blue)

        return image_array


class MixedDiskStimulusGenerator(GaussStimulusGenerator):

    def __init__(self, rect_size=(80,80), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var

    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        red = foreground_color[0]/255
        green = foreground_color[1]/255
        blue = foreground_color[2]/255

        stim_width, stim_height = self.image_size
        rect_width, rect_height = self.rect_size

        blob = self.multivariate_gaussian(self.rect_size)
        maximum = blob.max()
        blob = blob/(maximum)

        square_blob_dip = np.ones((self.rect_size[0], self.rect_size[1]))
        square_blob_dip -= blob

        #square_blob_dip *= blob.sum()/square_blob_dip.sum()

        x_coords, y_coords = np.meshgrid(np.arange(self.rect_size[0]),np.arange(self.rect_size[1]))
        x_coords = x_coords - self.rect_size[0]/2
        y_coords = y_coords - self.rect_size[1]/2
        distances = np.sqrt(np.power(x_coords,2) + np.power(y_coords,2))

        blob[distances > (self.rect_size[0]/2-1)] = 0.0;
        square_blob_dip[distances > (self.rect_size[0]/2-1)] = 0.0;

        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)

        image_array = np.ones((self.image_size[0], self.image_size[1],3), dtype=np.float32)*0.5
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += blob*(red-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += blob*(green-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += blob*(blue-0.5)

        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += square_blob_dip*(0.5-red)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += square_blob_dip*(0.5-green)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += square_blob_dip*(0.5-blue)

        return image_array


class RingCosStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,80), ring_thickness=0.1, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size

    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        red = foreground_color[0]/255
        green = foreground_color[1]/255
        blue = foreground_color[2]/255

        stim_width, stim_height = self.image_size
        rect_width, rect_height = self.rect_size

        x_coords, y_coords = np.meshgrid(np.arange(self.rect_size[0]),np.arange(self.rect_size[1]))
        x_coords = x_coords - self.rect_size[0]/2
        y_coords = y_coords - self.rect_size[1]/2
        distances = np.sqrt(np.power(x_coords,2) + np.power(y_coords,2))
        cos_base = (distances/(rect_width/2)) * np.pi * 0.5
        cos_base = np.cos(cos_base)
        cos_base[distances > rect_width/2] = 0.0

        stimulus = np.ones(self.rect_size)
        stimulus[distances > rect_width/2] = 0.0
        stimulus[distances < (rect_width/2)*(1-0.1)] = 0.0

        x_position = round((stim_width - rect_width) * stim_x_rel)
        y_position = round((stim_height - rect_height) * stim_y_rel)

        image_array = np.ones((self.image_size[0], self.image_size[1],3), dtype=np.float32)*0.5
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += cos_base*(0.5-red)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += cos_base*(0.5-green)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += cos_base*(0.5-blue)

        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,0] += stimulus*(red-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,1] += stimulus*(green-0.5)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width,2] += stimulus*(blue-0.5)

        return image_array
