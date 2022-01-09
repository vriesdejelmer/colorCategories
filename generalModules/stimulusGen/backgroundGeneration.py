#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 00:47:25 2021

@author: vriesdejelmer
"""

import random
from PIL import Image, ImageDraw, ImageFont
#import cv2

image_size = (224,224)
word = 'color'
font_size = 80

def getDrawerAndFont(font_type, background_color):
    img = Image.new('RGB', image_size, background_color)
    draw = ImageDraw.Draw(img)
    font_location = '/Users/vriesdejelmer/Dropbox/pythonProjects/categoricalAnalysis/data/fonts/another' + str(font_type) + '.ttf'
    fnt = ImageFont.truetype(font_location, font_size)
    return img, draw, fnt


def generateNewBackground():
    rand_back_grey = random.randint(80,175)

    background_color = (rand_back_grey, rand_back_grey, rand_back_grey)

    image, draw, fnt = getDrawerAndFont(random.randint(1,5), background_color)

    for idx in range(10):
        rand_grey = random.randint(0,255)
        text_size = draw.textsize(word, font=fnt)
        x_position = round((image_size[0] - text_size[0]) * random.uniform(0.0, 1.0))
        y_position = round((image_size[1] - text_size[1]) * random.uniform(0.0, 1.0))
        draw.text((x_position, y_position), word, fill=(rand_grey,rand_grey,rand_grey), font=fnt)

    return image

for idx in range(10000):
    if (idx % 1000) == 0:
        print(idx)

    image = generateNewBackground()
    image.save('/Users/vriesdejelmer/Dropbox/pythonProjects/categoricalAnalysis/data/backImages/f' + str(font_size) + '/bg_' + str(idx) + '.png')

