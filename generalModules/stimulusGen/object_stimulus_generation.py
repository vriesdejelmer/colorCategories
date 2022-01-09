#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:48:19 2020

@author: vriesdejelmer
"""

import numpy as np
import random
from .stimulus_generation import StimulusGenerator, MultiColorStimulusGenerator


class LineObjectStimulusGenerator(StimulusGenerator):

    def __init__(self, object_size=128, **kwargs):
        super().__init__(**kwargs)
        self.object_size = object_size

    def _getRGBStim(self, foreground_color, model_phase, object_type, index):
        object_coding = np.load('../data/lineDrawings/objects/' + object_type + '/' + model_phase + '/np_' + object_type + '_' + str(index) + '.npy')
        (r, g, b) = foreground_color
        object_image = np.empty(object_coding.shape, dtype=np.float32)
        object_image[object_coding[:,:,0]==1, 0] = self.background_color
        object_image[object_coding[:,:,1]==1, 1] = self.background_color
        object_image[object_coding[:,:,2]==1, 2] = self.background_color
        object_image[object_coding==-1] = self.background_color
        object_image[object_coding[:,:,0]==0, 0] = r/255.0
        object_image[object_coding[:,:,1]==0, 1] = g/255.0
        object_image[object_coding[:,:,2]==0, 2] = b/255.0

        x_rel_pos = random.uniform(0, 1)
        y_rel_pos = random.uniform(0, 1)

        width, height = self.image_size
        image_array = np.ones((width, height, 3), dtype=np.float32) * self.background_color
        x_position = round((width - self.object_size) * x_rel_pos)
        y_position = round((height - self.object_size) * y_rel_pos)
        image_array[y_position:y_position+self.object_size, x_position:x_position+self.object_size, :] = object_image

        return image_array


class ObjectStimulusGenerator:

    def generateObjectImage(self, foreground_color, model_phase, object_type, index, line_color):
        object_coding = np.load('../data/lineDrawings/objects/' + object_type + '/' + model_phase + '/np_' + object_type + '_' + str(index) + '.npy')
        (r, g, b) = foreground_color
        object_image = np.empty(object_coding.shape, dtype=np.float32)
        object_image[object_coding==0] = line_color-self.background_color
        object_image[object_coding[:,:,0]==1, 0] = r/255.0 - self.background_color
        object_image[object_coding[:,:,1]==1, 1] = g/255.0 - self.background_color
        object_image[object_coding[:,:,2]==1, 2] = b/255.0 - self.background_color
        object_image[object_coding==-1] = 0

        return object_image

    def drawObjectAtRandPos(self, image_array, object_image):
        x_rel_pos = random.uniform(0, 1)
        y_rel_pos = random.uniform(0, 1)

        width, height = self.image_size
        x_position = round((width - self.object_size) * x_rel_pos)
        y_position = round((height - self.object_size) * y_rel_pos)
        image_array[y_position:y_position+self.object_size, x_position:x_position+self.object_size, :] += object_image
        return image_array

class SingleObjectStimulusGenerator(StimulusGenerator, ObjectStimulusGenerator):

    def __init__(self, object_size=128, **kwargs):
        super().__init__(**kwargs)
        self.object_size = object_size

    def drawSingleObject(self, foreground_color, model_phase, object_type, index, line_color):
        object_image = self.generateObjectImage(foreground_color, model_phase, object_type, index, line_color=line_color)
        width, height = self.image_size
        image_array = np.ones((width, height, 3), dtype=np.float32) * self.background_color
        image_array = self.drawObjectAtRandPos(image_array, object_image)
        return image_array

    def _getRGBStim(self, foreground_color, model_phase, object_type, index):

        image_array = self.drawSingleObject(foreground_color, model_phase, object_type, index, line_color=0)
        return image_array

class SingleObjectStimulusGeneratorLumBackVar(SingleObjectStimulusGenerator):

    def drawSingleObject(self, foreground_color, model_phase, object_type, index, line_color):
        self.background_color = random.uniform(0,1.0)
        object_image = self.generateObjectImage(foreground_color, model_phase, object_type, index, line_color=line_color)
        width, height = self.image_size
        image_array = np.ones((width, height, 3), dtype=np.float32) * self.background_color
        image_array = self.drawObjectAtRandPos(image_array, object_image)
        return image_array
    
class SingleObjectStimulusGeneratorNoiseBack(SingleObjectStimulusGenerator):

    def generateObjectImage(self, foreground_color, model_phase, object_type, index, line_color):
        
        object_coding = np.load('../data/lineDrawings/objects/' + object_type + '/' + model_phase + '/np_' + object_type + '_' + str(index) + '.npy')
        (r, g, b) = foreground_color
        object_image = np.empty(object_coding.shape, dtype=np.float32)
        #object_image[object_coding==0] = line_color
        random_grey = random.uniform(0.0,1.0)
        object_image[object_coding[:,:,0]==0, 0] = random_grey
        object_image[object_coding[:,:,1]==0, 1] = random_grey
        object_image[object_coding[:,:,2]==0, 2] = random_grey
        
        object_image[object_coding[:,:,0]==1, 0] = r/255
        object_image[object_coding[:,:,1]==1, 1] = g/255
        object_image[object_coding[:,:,2]==1, 2] = b/255
        object_image[object_coding==-1] = -1
        
        return object_image


    def drawObjectAtRandPos(self, image_array, object_image):
        x_rel_pos = random.uniform(0, 1)
        y_rel_pos = random.uniform(0, 1)

        width, height = self.image_size
        x_position = round((width - self.object_size) * x_rel_pos)
        y_position = round((height - self.object_size) * y_rel_pos)
        
        image_array[y_position:y_position+self.object_size, x_position:x_position+self.object_size, :] = object_image

        return image_array    


    def drawSingleObject(self, foreground_color, model_phase, object_type, index, line_color):
        object_image = self.generateObjectImage(foreground_color, model_phase, object_type, index, line_color=line_color)
        width, height = self.image_size
        noise_image = np.random.rand(4,4)
        noise_image = noise_image.repeat(56, axis=0).repeat(56, axis=1)
        noise_image = np.array([noise_image, noise_image, noise_image]).transpose(1,2,0)
        image_array = np.ones((width, height, 3), dtype=np.float32) * -1
        image_array = self.drawObjectAtRandPos(image_array, object_image)
        image_array[image_array==-1] = noise_image[image_array==-1]
        return image_array

class ObjectNoLineStimulusGenerator(SingleObjectStimulusGenerator):

    def __init__(self, object_size=128, **kwargs):
        super().__init__(**kwargs)
        self.object_size = object_size

    def _getRGBStim(self, foreground_color, model_phase, object_type, index):

        image_array = self.drawSingleObject(foreground_color, model_phase, object_type, index, line_color=self.background_color)
        return image_array


class DoubleObjectStimulusGenerator(MultiColorStimulusGenerator, ObjectStimulusGenerator):

    def __init__(self, object_size=128, **kwargs):
        super().__init__(**kwargs)
        self.object_size = object_size


    def placeObjectOnImage(self, image_array, x_rel_pos, y_rel_pos, object_image):
        width, height = self.image_size
        x_position = round((width - self.object_size) * x_rel_pos)
        y_position = round((height - self.object_size) * y_rel_pos)
        image_array[y_position:y_position+self.object_size, x_position:x_position+self.object_size, :] += object_image


    def _getRGBStim(self, foreground_color, model_phase, object_types, indices):

        object_image1 = self.generateObjectImage(foreground_color[0], model_phase, object_types[0], indices[0], line_color=0)
        object_image2 = self.generateObjectImage(foreground_color[1], model_phase, object_types[1], indices[1], line_color=0)

        x_rel_pos1 = random.uniform(0, 1)
        y_rel_pos1 = random.uniform(0, 1)
        if x_rel_pos1 > 0.5: x_rel_pos2 = random.uniform(0, 0.4)
        else: x_rel_pos2 = random.uniform(0.6, 1)

        if y_rel_pos1 > 0.5: y_rel_pos2 = random.uniform(0, 0.3)
        else: y_rel_pos2 = random.uniform(0.7, 1)


        width, height = self.image_size
        image_array = np.zeros((width, height, 3), dtype=np.float32)

        self.placeObjectOnImage(image_array, x_rel_pos1, y_rel_pos1, object_image1)
        self.placeObjectOnImage(image_array, x_rel_pos2, y_rel_pos2, object_image2)

        image_array += self.background_color

        image_array[image_array < 0] = 0
        image_array[image_array > 1.0] = 1.0

        return image_array
