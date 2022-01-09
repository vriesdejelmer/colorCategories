    #import external libraries
import os, platform, sys
import torch
from pathlib import Path

    #set cwd and paths
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir).parent))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent.parent, 'generalModules')))

    #import local functions
from model_creation import getModels, saveModels
from model_evaluation import findHueClassification
import torch.optim as optim
from training_functions import trainModels
from data_management import DatasetManagement, DatasetProperties
from stimulusGen.hue_conversion import LabColor, HSVColor, RGBHueCircle
from stimulusGen.word_stimulus_generation import SingleWordStimulusGenerator
from datasets.hue_word_datasets import WordColorDataset


    #property object for learning
class WordColorProperties(DatasetProperties):

    def __init__(self, words, hue_ranges, **kwargs):
        super().__init__(**kwargs)
        self.words = words
        self.hue_ranges = hue_ranges

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #analysis parameters
samples_per_class = {'train': 500, 'test': 50}
font_size = 40
num_epochs = 30
brightness = 1.0
learning_rate = 0.001
pretrained = False
batch_size = 16

stim_gen = SingleWordStimulusGenerator(font_size=font_size, color_convertor=HSVColor(brightness))

words = ['color', 'color', 'color', 'color', 'color', 'color', 'color'] # 'car'
hue_ranges = [(0.9, 0.04), (0.04, 0.12), (0.12, 0.24), (0.24, 0.43), (0.43, 0.57), (0.57, 0.78), (0.78, 0.9)]

data_props = WordColorProperties(words, hue_ranges, batch_size=batch_size, samples_per_class=samples_per_class)

    #output folder
model_folder = '../models/'

data_manager = DatasetManagement(WordColorDataset, data_props, stimulus_generator=stim_gen)

frozen_models = getModels('resnet18', 1, len(words), pretrained=pretrained, frozen=pretrained)
model_list = [(model_index, model, optim.SGD(model.parameters(), lr=learning_rate)) for model_index, model in enumerate(frozen_models)]
model_list, _ = trainModels(model_list, data_manager, num_epochs, device)

saveModels(model_list, model_folder, 0, 'categorical_f40')