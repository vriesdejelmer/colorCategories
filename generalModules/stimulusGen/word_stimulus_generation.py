#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:48:19 2020

Stimulus generation extensions for generating word stimuli

@author: vriesdejelmer
"""

import random
from PIL import Image, ImageDraw, ImageFont
import string
#import cv2
import numpy as np
from torchvision import transforms
from .stimulus_generation import StimulusGenerator, MultiColorStimulusGenerator

class WordStimulusGenerator:

    def getDrawerAndFont(self, font_type):
        img = Image.new('RGB', self.image_size, self.background_color)
        draw = ImageDraw.Draw(img)
        font_location = '../data/fonts/another' + str(font_type) + '.ttf'
        fnt = ImageFont.truetype(font_location, self.font_size)
        return img, draw, fnt

class SingleWordStimulusGenerator(StimulusGenerator, WordStimulusGenerator):

    def __init__(self, default_word='Color', font_size=40, font_types=[1,2,3,4,5], **kwargs):
        super().__init__(**kwargs)
        self.default_word = default_word
        self.font_size = font_size
        self.font_types = font_types


    def _getRGBStim(self, foreground_color, text_x_rel, text_y_rel, font_type, word=None):
        if word == None:
            word = self.default_word

        image, draw, fnt = self.getDrawerAndFont(font_type)

        text_size = draw.textsize(word, font=fnt)
        x_position = round((self.image_size[0] - text_size[0]) * text_x_rel)
        y_position = round((self.image_size[1] - text_size[1]) * text_y_rel)
        draw.text((x_position, y_position), word, fill=foreground_color, font=fnt)

        return image


class SingleWordSavedBackgroundStimulusGenerator(StimulusGenerator, WordStimulusGenerator):

    def __init__(self, default_word='Color', font_size=60, font_types=[1,2,3,4,5], back_word_count=10, **kwargs):
        super().__init__(**kwargs)
        self.default_word = default_word
        self.back_word_count = back_word_count
        self.font_size = font_size
        self.font_types = font_types


    def _getRGBStim(self, foreground_color, text_x_rel, text_y_rel, font_type, word=None):
        if word == None:
            word = self.default_word

        font_type = random.randint(1,5)
        idx = random.randint(0,9999)

        with Image.open('../data/backImages/f' + str(self.font_size) + '/bg_' + str(idx) + '.png') as img:
            draw = ImageDraw.Draw(img)
            font_location = '../data/fonts/another' + str(font_type) + '.ttf'
            fnt = ImageFont.truetype(font_location, self.font_size)

            text_size = draw.textsize(word, font=fnt)
            x_position = round((self.image_size[0] - text_size[0]) * text_x_rel)
            y_position = round((self.image_size[1] - text_size[1]) * text_y_rel)
            draw.text((x_position, y_position), word, fill=foreground_color, font=fnt)

            return img


class SingleWordMultiBackgroundStimulusGenerator(StimulusGenerator, WordStimulusGenerator):

    def __init__(self, default_word='Color', font_size=60, font_types=[1,2,3,4,5], back_word_count=10, **kwargs):
        super().__init__(**kwargs)
        self.default_word = default_word
        self.back_word_count = back_word_count
        self.font_size = font_size
        self.font_types = font_types


    def _getRGBStim(self, foreground_color, text_x_rel, text_y_rel, font_type, word=None):
        if word == None:
            word = self.default_word

        rand_back_grey = random.randint(80,175)

        self.background_color = (rand_back_grey, rand_back_grey, rand_back_grey)

        image, draw, fnt = self.getDrawerAndFont(font_type)

        for idx in range(self.back_word_count):
            rand_grey = random.randint(0,255)
            text_size = draw.textsize(word, font=fnt)
            x_position = round((self.image_size[0] - text_size[0]) * random.uniform(0.0, 1.0))
            y_position = round((self.image_size[1] - text_size[1]) * random.uniform(0.0, 1.0))
            draw.text((x_position, y_position), word, fill=(rand_grey,rand_grey,rand_grey), font=fnt)

        text_size = draw.textsize(word, font=fnt)
        x_position = round((self.image_size[0] - text_size[0]) * text_x_rel)
        y_position = round((self.image_size[1] - text_size[1]) * text_y_rel)
        draw.text((x_position, y_position), word, fill=foreground_color, font=fnt)

        return image


class SingleWorStimGenLab(SingleWordStimulusGenerator):

    def __init__(self, **kwargs):
       super().__init__(**kwargs)

       mean = np.array([0.5, 0.5, 0.5])
       std = np.array([0.25, 0.25, 0.25])
       self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) #transforms.RandomHorizontalFlip(),


    def getStimImage(self, foreground_color, **kwargs):

        rgb_color = self.convertToRGB(foreground_color)
        image = self._getRGBStim(rgb_color, **kwargs)

        image = np.asarray(image).copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        if not self.transform is None:
            image = self.transform(image)

        return image


class TwoWordStimulusGenerator(MultiColorStimulusGenerator, WordStimulusGenerator):

    def __init__(self, font_size=80, font_types=[1,2,3,4,5], **kwargs):
        super().__init__(**kwargs)
        self.font_size = font_size
        self.font_types = font_types

    def getTwoWordImage(self, fg_colors, words):

        [font_type] = random.sample(self.font_types, 1)
        img, draw, fnt = self.getDrawerAndFont(font_type)

        for fg_color, word in zip(fg_colors, words):
            text_size = draw.textsize(word, font=fnt)
            x_position = round((self.image_size[0] - text_size[0]) * random.uniform(0, 1))
            y_position = round((self.image_size[1] - text_size[1]) * random.uniform(0, 1))
            draw.text((x_position, y_position), word, fill=fg_color, font=fnt)

        return img


    def _getRGBStim(self, foreground_colors, word_combo, background_color=None):
        if not (background_color is None):
            self.background_color = background_color

        image = self.getTwoWordImage(foreground_colors, word_combo)

        return image


class ClutteredWordStimulusGenerator(MultiColorStimulusGenerator, WordStimulusGenerator):

    def __init__(self, default_word='Color', font_size=80, font_types=[1,2,3,4,5], **kwargs):
        super().__init__(**kwargs)
        self.default_word = default_word
        self.font_size = font_size
        self.font_types = font_types


    def getRandomString(self, length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str #.capitalize()


    def getMultiWordImage(self, fg_colors, main_word):

        [font_type] = random.sample(self.font_types, 1)
        img, draw, fnt = self.getDrawerAndFont(font_type)

        for index, fg_color in enumerate(fg_colors):

                #first draw class words, then distractor words
            if index < 3:
                word = main_word
            else:
                word = 'color'

            text_size = draw.textsize(word, font=fnt)
            x_position = round((self.image_size[0] - text_size[0]) * random.uniform(0, 1))
            y_position = round((self.image_size[1] - text_size[1]) * random.uniform(0, 1))

            draw.text((x_position, y_position), word, fill=fg_color, font=fnt)

        return img


    def _getRGBStim(self, foreground_colors, word, background_color=None):
        if not (background_color is None):
            self.background_color = background_color

        image = self.getMultiWordImage(foreground_colors, word)

        return image
