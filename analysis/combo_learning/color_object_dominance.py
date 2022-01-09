    #import external libraries
import os, sys
from pathlib import Path
import numpy as np
import csv

    #set cwd and paths
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir).parent))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent.parent, 'generalModules')))

    #import local functions
from model_creation import getModels
from model_evaluation import findHueClassification
import torch.optim as optim
from training_functions import trainModels
from data_management import RangeDatasetManagement
from stimulusGen.hue_conversion import HSVColor
from stimulusGen.object_stimulus_generation import SingleObjectStimulusGenerator
from datasets.doodle_datasets import ObjectColorDataset, ObjectHueRangeDataset
from general_functions import writeProperties, getDevice

    
    #object holding learning props
class ColoredObjectProperties:

    def __init__(self, object_types, object_hues, batch_size, samples_per_class, samples_per_color, num_workers):
        self.samples_per_color = samples_per_color
        self.object_types = object_types
        self.object_hues = object_hues
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.num_workers = num_workers

    
    #function for generating hue ranges
def getDoubleHueRanges(original_borders):
    hue_ranges = []
    for index in range(len(original_borders)):
        next_index = (index+1) % len(original_borders)
        category_range = original_borders[next_index] - original_borders[index]

        if category_range < 0: category_range += 1

        left_range = (original_borders[index]+(category_range*0.2), original_borders[index]+(category_range*0.4))
        right_range = (original_borders[next_index]-(category_range*0.4), original_borders[next_index]-(category_range*0.2))

        hue_ranges.append(left_range)
        hue_ranges.append(right_range)
    return hue_ranges


device = getDevice()

old_vars = vars().copy()

    #analysis parameters
num_epochs = 8
repetitions = 100
brightness = 1.0
learning_rate = 0.001
batch_size = 8
pretrained = True
samples_per_class={'train': 500, 'test': 50}
samples_per_color = 80
data_folder = '../data/objectColor/twoObjectsPerCatFinal/'

original_borders = np.array([3, 10, 23, 37, 56, 78, 88])/100
stim_gen = SingleObjectStimulusGenerator(color_convertor=HSVColor(brightness), background_color=128)
object_types = ['angel', 'apple', 'axe', 'basketball', 'carrot', 'cloud', 'cow', 'crab', 'crown', 'dolphin', 'duck', 'mushroom', 'sailboat', 'school bus']
hue_ranges = getDoubleHueRanges(original_borders)

writeProperties(vars().copy(), old_vars, data_folder)

for rep in range(repetitions):

    print('Starting rep ' + str(rep) + '......')

    objects = np.random.permutation(object_types)
    print(objects)

    data_props = ColoredObjectProperties(objects, hue_ranges, batch_size=batch_size, samples_per_class=samples_per_class, samples_per_color=samples_per_color, num_workers=4)

        #output folder
    data_folder_rep = data_folder + str(rep) + '/'
    if not os.path.exists(data_folder_rep):
        os.mkdir(data_folder_rep)

    with open(data_folder_rep + 'object_list.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(objects)
        wr.writerow(hue_ranges)

    data_manager = RangeDatasetManagement(ObjectColorDataset, ObjectHueRangeDataset, data_props, stimulus_generator=stim_gen)

    frozen_models = getModels('resnet18', 1, len(object_types), pretrained=pretrained, frozen=pretrained)
    model_list = [(model_index, model, optim.SGD(model.parameters(), lr=learning_rate)) for model_index, model in enumerate(frozen_models)]
    model_list, _ = trainModels(model_list, data_manager, num_epochs, device)

    for index, object_type in enumerate(objects):
        findHueClassification([model for _, model, _ in model_list], data_manager, data_folder_rep, index, object_type)

