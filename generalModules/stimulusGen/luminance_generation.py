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
import cv2
import imageio

class LuminanceStimulusGenerator():

    def __init__(self, color_convertor=None, transform=None, image_size=(224,224), background_color=(128,128,128)):
        self.background_color = background_color
        self.image_size = image_size
        self.color_convertor = color_convertor

        if transform == None:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) #transforms.RandomHorizontalFlip(),
        else:
            self.transform = transform

    def getStimImage(self, color1, color2,  **kwargs):

        image = self._getRGBStim(color1, color2, **kwargs)
        if not self.transform is None:
            image = self.transform(image)
        
        return image



class FigureGroundStimulusGenerator(LuminanceStimulusGenerator):

    def _getRGBStim(self, color1, color2, **kwargs):

        im = imageio.imread('../data/stimMaterial/figureground.png')
        im_size = im.shape
        im_ratio = im_size[1]/im_size[0]
        im = cv2.resize(im, (int(224*im_ratio), 224))
        np_array = np.asarray(im)
        np_array = np_array[0:224, 0:224]
        foreground_array = np_array[:,:,0]/255
        background_array = 1 - np_array[:,:,0]/255

        new_image = np.zeros((224, 224, 3), dtype=np.float32)
        new_image[:,:, 0] += foreground_array * color1[0]
        new_image[:,:, 1] += foreground_array * color1[1]
        new_image[:,:, 2] += foreground_array * color1[2]

        new_image[:,:,0] += background_array * color2[0]
        new_image[:,:,1] += background_array * color2[1]
        new_image[:,:,2] += background_array * color2[2]
        return new_image


class SideBySideStimulusGenerator(LuminanceStimulusGenerator):

    def _getRGBStim(self, color1, color2, **kwargs):

        new_image = np.ones((224, 224, 3), dtype=np.float32)*0.5
        
        new_image[40:112,40:184,0] = color1[0]
        new_image[40:112,40:184,1] = color1[1]
        new_image[40:112,40:184,2] = color1[2]

        new_image[112:184,40:184,0] = color2[0]
        new_image[112:184,40:184,1] = color2[1]
        new_image[112:184,40:184,2] = color2[2]
        return new_image