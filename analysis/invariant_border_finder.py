    #import external libraries
import os, sys
from pathlib import Path

    #set cwd and path
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

    #import local functions
from model_creation import getModels, saveModels
import torch.optim as optim
from model_evaluation import findHueClassification
from training_functions import trainModels
from data_management import RangeDatasetManagement, InvariantDatasetProperties
from stimulusGen.hue_conversion import HSVColor #HSVColorBrightRange #LabColor, HSVColor, RGBHueCircle,
from stimulusGen.stimulus_generation import getCentralHueMatrix
from datasets.hue_word_datasets import FocalColorDataset, HueRangeDataset
from stimulusGen.word_stimulus_generation import SingleWordStimulusGenerator
from general_functions import writeProperties, getDevice


old_vars = vars().copy()    #log current working mem so we can later save local variables

device = getDevice()

    #output folder
folder_name = 'objectClass_f40'
data_folder = '../data/invariantBorders/' + folder_name + '/'
model_folder = '../models/invariantBorders/' + folder_name + '/'


    #analysis parameters
network_type = 'resnet18'
samples_per_class = {'train': 500, 'test': 50}
model_count = 150
total_range = 0.05
num_epochs = 5
batch_size = 8
brightness = 1.0
samples_per_color = 60
font_size = 40
learning_rate = 0.001
parts_lists = [7, 6, 8, 5, 4, 9]
pretrained = True   #if True models weight are also frozen
save_models = False     #if true save the trained models
color_convertor = HSVColor(brightness)
stimulus_generator = SingleWordStimulusGenerator

writeProperties(vars().copy(), old_vars, data_folder)   #save local variables

    #initialize the stimulus generaotr and the learning property object
stim_gen = stimulus_generator(color_convertor=color_convertor, font_size=font_size)
data_props = InvariantDatasetProperties(batch_size=batch_size, samples_per_class=samples_per_class, samples_per_color=samples_per_color)

    
for parts in parts_lists:
    print('Running for ' + str(parts) + ' parts')

        #set parts specific learning_props
    data_props.focal_hues_matrix = getCentralHueMatrix(parts, model_count)
    data_props.hue_range = total_range / parts   #determine the range of the hues

        #initialize a dataset manager
    data_manager = RangeDatasetManagement(FocalColorDataset, HueRangeDataset, data_props, stimulus_generator=stim_gen)
        
        #load and train models
    frozen_models = getModels(network_type, model_count, parts, pretrained=pretrained, frozen=pretrained)
    model_list = [(model_index, model, optim.SGD(model.parameters(), lr=learning_rate)) for model_index, model in enumerate(frozen_models)]
    model_list, _ = trainModels(model_list, data_manager, num_epochs, device)

    if save_models: #save models if desired
        saveModels(model_list, model_folder, parts)

        #analyze classifcation after training a specific range
    findHueClassification(frozen_models, data_manager, data_folder, parts)
