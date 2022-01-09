#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:48:19 2020

For creating illusion stimuli. Needs rewrite due to refactoring

@author: vriesdejelmer
"""

import numpy as np
import random
from .stimulus_generation import StimulusGenerator


class YellowOnBlueIllusionStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,50), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var


    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        width, height = self.image_size
        rect_width, rect_height = self.rect_size
        image_array = np.ones((width, height, 3), dtype=np.float32) * np.array([0.0, 0.0, 0.8], dtype = np.float32)
        dot_color = np.array([0.8, 0.8, 0.0], dtype = np.float32)

        x_position = round((width - rect_width) * stim_x_rel)
        y_position = round((height - rect_height) * stim_y_rel)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width, :] = [color_int/255 for color_int in foreground_color]

        for x_index in range(1,16):
            for y_index in range(1,16):
                image_array[y_index*14-4:y_index*14+4, x_index*14-4:x_index*14+4] = dot_color

        return image_array


class BlueOnYellowIllusionStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,50), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var


    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        width, height = self.image_size
        rect_width, rect_height = self.rect_size
        image_array = np.ones((width, height, 3), dtype=np.float32) * np.array([.8, .8, 0.0], dtype = np.float32)
        dot_color = np.array([0.0, 0.0, 0.8], dtype = np.float32)

        x_position = round((width - rect_width) * stim_x_rel)
        y_position = round((height - rect_height) * stim_y_rel)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width, :] = [color_int/255 for color_int in foreground_color]

        for x_index in range(1,16):
            for y_index in range(1,16):
                image_array[y_index*14-4:y_index*14+4, x_index*14-4:x_index*14+4] = dot_color

        return image_array


class BWGridStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,50), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var


    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        if random.random() >0.5:
            back_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            dot_color = np.array([0.8,0.8,0.8], dtype=np.float32)
        else:
            back_color = np.array([0.8, 0.8, 0.8], dtype=np.float32)
            dot_color = np.array([0.0,0.0,0.0], dtype=np.float32)

        width, height = self.image_size
        rect_width, rect_height = self.rect_size

        image_array = np.ones((width, height, 3), dtype=np.float32) * back_color

        x_position = round((width - rect_width) * stim_x_rel)
        y_position = round((height - rect_height) * stim_y_rel)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width, :] = [color_int/255 for color_int in foreground_color]

        for x_index in range(1,16):
            for y_index in range(1,16):
                image_array[y_index*14-4:y_index*14+4, x_index*14-4:x_index*14+4] = dot_color

        image_array[image_array < 0] = 0.0
        image_array[image_array > 1] = 1.0

        return image_array


class GridStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,50), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var


    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        width, height = self.image_size
        rect_width, rect_height = self.rect_size
        
        back_hue = random.uniform(0,1)
        front_hue = back_hue + 0.5

        back_color = np.array(self.color_convertor.toIntRGB(back_hue, brightness=0.8), dtype=np.float32)/255.0
        image_array = np.ones((width, height, 3), dtype=np.float32) * back_color

        x_position = round((width - rect_width) * stim_x_rel)
        y_position = round((height - rect_height) * stim_y_rel)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width, :] = [color_int/255 for color_int in foreground_color]

        dot_color = np.array(self.color_convertor.toIntRGB(front_hue, brightness=0.8), dtype=np.float32)/255.0
        for x_index in range(1,16):
            for y_index in range(1,16):
                image_array[y_index*14-4:y_index*14+4, x_index*14-4:x_index*14+4] = dot_color

        image_array[image_array < 0] = 0.0
        image_array[image_array > 1] = 1.0

        return image_array

class FullRandomGridStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,50), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var


    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        width, height = self.image_size
        rect_width, rect_height = self.rect_size
        rect_height = round(rect_height + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_height)
        rect_width = round(rect_width + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_width)

        back_hue = random.uniform(0,1)
        front_hue = random.uniform(0,1)

        back_color = np.array(self.color_convertor.toIntRGB(back_hue, brightness=0.8), dtype=np.float32)/255.0
        image_array = np.ones((width, height, 3), dtype=np.float32) * back_color

        x_position = round((width - rect_width) * stim_x_rel)
        y_position = round((height - rect_height) * stim_y_rel)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width, :] = [color_int/255 for color_int in foreground_color]

        dot_color = np.array(self.color_convertor.toIntRGB(front_hue, brightness=0.8), dtype=np.float32)/255.0
        for x_index in range(1,16):
            for y_index in range(1,16):
                image_array[y_index*14-4:y_index*14+4, x_index*14-4:x_index*14+4] = dot_color

        image_array[image_array < 0] = 0.0
        image_array[image_array > 1] = 1.0

        return image_array

class RedOnGreenIllusionStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,50), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var


    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        width, height = self.image_size
        rect_width, rect_height = self.rect_size
        rect_height = round(rect_height + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_height)
        rect_width = round(rect_width + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_width)
        image_array = np.ones((width, height, 3), dtype=np.float32) * np.array([0.0, 0.8, 0.0], dtype = np.float32)
        dot_color = np.array([0.8, 0.0, 0.0], dtype = np.float32)

        x_position = round((width - rect_width) * stim_x_rel)
        y_position = round((height - rect_height) * stim_y_rel)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width, :] = [color_int/255 for color_int in foreground_color]

        for x_index in range(1,16):
            for y_index in range(1,16):
                image_array[y_index*14-4:y_index*14+4, x_index*14-4:x_index*14+4] = dot_color

        return image_array


class GreenOnRedIllusionStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(80,50), rect_size_var=0.25, **kwargs):
        super().__init__(**kwargs)
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var


    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel):

        width, height = self.image_size
        rect_width, rect_height = self.rect_size
        rect_height = round(rect_height + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_height)
        rect_width = round(rect_width + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_width)
        image_array = np.ones((width, height, 3), dtype=np.float32) * np.array([.8, .0, 0.0], dtype = np.float32)
        dot_color = np.array([0.0, 0.8, 0.0], dtype = np.float32)

        x_position = round((width - rect_width) * stim_x_rel)
        y_position = round((height - rect_height) * stim_y_rel)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width, :] = [color_int/255 for color_int in foreground_color]

        for x_index in range(1,16):
            for y_index in range(1,16):
                image_array[y_index*14-4:y_index*14+4, x_index*14-4:x_index*14+4] = dot_color

        return image_array


