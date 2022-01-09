import os, sys
import torch
import random
import numpy as np
from pathlib import Path

#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir).parent))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent.parent, 'generalModules')))

    #import local functions
from model_creation import getModels
from model_evaluation import findHueClassification
import torch.optim as optim
from training_functions import trainModels
from data_management import RangeDatasetManagement, InvariantDatasetProperties
from stimulusGen.hue_conversion import LabColor, HSVColor, RGBHueCircle
from datasets.hue_word_datasets import FocalMultiColorDataset, ColorMultiRangeDataset
from stimulusGen.word_stimulus_generation import ClutteredWordStimulusGenerator
from general_functions import writeProperties, getDevice

class MultiWordColorProperties(InvariantDatasetProperties):

    def __init__(self, words, focal_hues, **kwargs):
        super().__init__(**kwargs)
        self.words = words
        self.focal_hues = focal_hues


def getCategorySelection(original_borders, splits, cat_prop):

    hue_ranges = []
    for index in range(len(original_borders)):
        next_index = (index+1) % len(original_borders)
        left_border = original_borders[index]
        right_border = original_borders[next_index]
        if right_border < left_border:
            right_border += 1

        range_start = left_border + (right_border - left_border) * cat_prop
        hue_range = (right_border-left_border)/splits
        hue_ranges.append((range_start, range_start+hue_range))

    return hue_ranges

old_vars = vars().copy()

device = getDevice()

    #analysis parameters

original_borders = np.array([3, 10, 23, 37, 56, 78, 88])/100

num_epochs = 10
brightness = 1.0
learning_rate = 0.001
pretrained = True
iterations = 15
samples_per_class={'train': 500, 'test': 50}
cat_splits = 10
font_size = 40
stim_gen = ClutteredWordStimulusGenerator(color_convertor=HSVColor(brightness), font_size=font_size)
samples_per_color = 100

words = ['color', 'color', 'color', 'color', 'color', 'color', 'color']

data_folder = '../data/wordColor/multiBorderResultFinalCats_f40/'

writeProperties(vars().copy(), old_vars, data_folder)

for spl_index, start_prop in enumerate(np.linspace(0,1-(1/cat_splits), cat_splits)):

    for index in range(iterations):

        hues = getCategorySelection(original_borders, cat_splits, start_prop)

        data_props = MultiWordColorProperties(words, hues, batch_size=4, samples_per_class=samples_per_class, samples_per_color=samples_per_color)

        data_manager = RangeDatasetManagement(FocalMultiColorDataset, ColorMultiRangeDataset, data_props, stimulus_generator=stim_gen)

        frozen_models = getModels('resnet18', 1, len(words), pretrained=pretrained, frozen=pretrained)
        model_list = [(model_index, model, optim.SGD(model.parameters(), lr=learning_rate)) for model_index, model in enumerate(frozen_models)]
        model_list, _ = trainModels(model_list, data_manager, num_epochs, device)

        findHueClassification([model for _, model, _ in model_list], data_manager, data_folder, int((spl_index+1)*100 + index), 'color')
