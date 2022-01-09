#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:48:19 2020

Base classes for stimulus generation and extension for generating colored rects

@author: vriesdejelmer
"""

import numpy as np
from torchvision import transforms
import random

class StimulusGenerator():

    def __init__(self, color_convertor=None, transform=None, image_size=(224,224), background_color=(128,128,128)):
        self.background_color = background_color
        self.image_size = image_size
        self.color_convertor = color_convertor

        if transform == None:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        else:
            self.transform = transform


    def getStimImage(self, foreground_color, **kwargs):

        rgb_color = self.convertToRGB(foreground_color)
        image = self._getRGBStim(rgb_color, **kwargs)

        if not self.transform is None:
            image = self.transform(image)

        return image


    def convertToRGB(self, foreground_color):
        if self.color_convertor != None:
            return self.color_convertor.toIntRGB(foreground_color)

        return foreground_color


class MultiColorStimulusGenerator(StimulusGenerator):

    def convertToRGB(self, foreground_color):
        if self.color_convertor != None:
            converted_colors = []
            for color in foreground_color:
                converted_colors.append(self.color_convertor.toIntRGB(color))

            return converted_colors

        return foreground_color


class RectStimulusGenerator(StimulusGenerator):

    def __init__(self, rect_size=(60,30), rect_size_var=0.25, **kwargs):
        self.rect_size = rect_size
        self.rect_size_var = rect_size_var


    def _getRGBStim(self, foreground_color, stim_x_rel, stim_y_rel, font_type):
        width, height = self.image_size

        rect_width, rect_height = self.rect_size
        rect_height = round(rect_height + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_height)
        rect_width = round(rect_width + random.uniform(-self.rect_size_var, self.rect_size_var) * rect_width)

        image_array = np.ones((width, height, 3), dtype=np.float32) * 0.5
        x_position = round((width - rect_width) * stim_x_rel)
        y_position = round((height - rect_height) * stim_y_rel)
        image_array[y_position:y_position+rect_height, x_position:x_position+rect_width, :] = [color_int/255 for color_int in foreground_color]

        return image_array


def getCentralHueMatrix(parts, model_count):
    return np.array([[(x/parts+y/(parts*model_count)) for x in range(parts)] for y in range(model_count)])
