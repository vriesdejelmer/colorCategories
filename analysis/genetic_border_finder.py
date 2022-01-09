    #import external libraries
import numpy as np
import os, sys
import torch.optim as optim
from pathlib import Path


    #set cwd and path
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))


    # import local functions
from offspring_creation import getNextGeneration, initializeNewBorderMatrix, initializeBorderMatrix
from training_functions import trainModels
from model_creation import getModels
from stimulusGen.word_stimulus_generation import SingleWordStimulusGenerator
from stimulusGen.hue_conversion import HSVColor
from datasets.hue_word_datasets import BoundaryHueDataset
from data_management import DatasetManagement, GenDatasetProperties
from general_functions import writeProperties, getDevice

device = getDevice()

data_folder = '../data/geneticBorders1/collapse/'
if len(sys.argv) > 1:
    data_folder += sys.argv[1] + '/'
            #output folder
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

old_vars = vars().copy()    #log current working mem for detecting local variables later

    #initialize local variables
model_count = 100
iterations = 40
learning_rate = 0.001
parts = 7
from_scratch = True #start from scratch or build on previously found borders?
pretrained = True
num_epochs = 3
batch_size = 8
mutation_prop = 0.025
distance_threshold = 0.00
font_size = 40
mutation_scale = 0.025
cross_method = 'collapse' #crossover, recombine, collapse
distribution = [0.55, 0.3, 0.15, 0.0]
samples_per_class = {'train': 500, 'test': 50}

writeProperties(vars().copy(), old_vars, data_folder)   #save newly initialized variables

    #if we are going from scratch we initilize a random border matrix
if from_scratch:
    border_matrix = initializeNewBorderMatrix(model_count, parts)
else:   #else we load excisting
    border_matrix = initializeBorderMatrix(model_count, parts, data_folder)


    #set up learning props, stimulus generator and dataset manager
data_props = GenDatasetProperties(border_matrix, batch_size=batch_size, samples_per_class=samples_per_class, num_workers=2)
stim_gen = SingleWordStimulusGenerator(color_convertor=HSVColor(), font_size=font_size)
data_manager = DatasetManagement(BoundaryHueDataset, data_props, stimulus_generator=stim_gen)


    #run evolutionary algorithm
for iter_index in range(iterations):
    print('----- Generation ' + str(iter_index))
    frozen_models = getModels('resnet18', model_count, parts, pretrained=pretrained, frozen=pretrained)
    model_list = [(model_index, model, optim.SGD(model.parameters(), lr=learning_rate)) for model_index, model in enumerate(frozen_models)]
    model_list, accuracies = trainModels(model_list, data_manager, num_epochs, device)

    if iter_index >= (iterations - 10):
        print('no mut')
        data_props.border_matrix = getNextGeneration(accuracies, data_props.border_matrix, top_count=10, mutation_prop=0.0, distribution=distribution, distance_threshold=distance_threshold, mutation_scale=mutation_scale, cross=cross_method)
    else:
        print('with mut')
        data_props.border_matrix = getNextGeneration(accuracies, data_props.border_matrix, top_count=10, mutation_prop=mutation_prop, distribution=distribution, distance_threshold=distance_threshold, mutation_scale=mutation_scale, cross=cross_method)

    np.save(data_folder + 'genetic_border_matrix_' + str(parts) + 'parts.npy', data_props.border_matrix)
    np.save(data_folder + 'genetic_border_matrix_' + str(iter_index), data_props.border_matrix)
