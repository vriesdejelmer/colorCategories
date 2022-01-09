#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 21:48:03 2020

@author: vriesdejelmer
"""
import torch
from stimulusGen.stimulus_generation import StimulusGenerator


class DatasetManagement:

    def __init__(self, train_dataset, data_properties, **kwargs):

            #we want a matrix with all hues (each row holds hue for iteration)
        self.data_props = data_properties
        self.train_dataset = train_dataset

        self.initializeStimGenerator(**kwargs)


    def initializeStimGenerator(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'stimulus_generator':
                self.stim_gen = value
            elif key == 'color_conversion':
                self.stim_gen = StimulusGenerator(color_convertor=value)

        if not hasattr(self,'stim_gen'):
            self.stim_gen = StimulusGenerator()


    def getDatasetsForModel(self, index=0, num_workers=0):

        print(self.stim_gen)
        image_datasets = {'train': self.train_dataset('train', self.data_props, index, stimulus_generator=self.stim_gen),
                          'test': self.train_dataset('test', self.data_props, index, stimulus_generator=self.stim_gen)}

        data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.data_props.batch_size,
                                                       shuffle=True, num_workers=self.data_props.num_workers) for x in ['train', 'test']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

        return data_loaders, dataset_sizes


    def getFocalHues(self, idx):
        if hasattr(self.data_props, 'focal_hues_matrix'):
            if self.data_props.focal_hues_matrix.ndim == 1:
                return self.data_props.focal_hues_matrix
            elif self.data_props.focal_hues_matrix.ndim == 2:
                return self.data_props.focal_hues_matrix[idx]
        else:
            return None


    def getHueConvertor(self):
        return self.stim_gen.color_convertor


class RangeDatasetManagement(DatasetManagement):

    def __init__(self, train_dataset, evaluate_dataset, data_properties, **kwargs):
        super().__init__(train_dataset, data_properties, **kwargs)
            #we want a matrix with all hues (each row holds hue for iteration)
        self.evaluate_dataset = evaluate_dataset


    def getFullRangeLoader(self, range_properties):
                #we create a data_loader that goes through the full spectrum

        full_range_dataset = self.evaluate_dataset(range_properties, self.data_props.samples_per_color, stimulus_generator=self.stim_gen)
        full_range_loader = torch.utils.data.DataLoader(full_range_dataset, batch_size=self.data_props.samples_per_color, num_workers=self.data_props.num_workers)
        return full_range_loader


    def getSamplesPerColor(self):
        return self.data_props.samples_per_color


class DatasetProperties:

    def __init__(self, batch_size, samples_per_class, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_class = samples_per_class


class InvariantDatasetProperties(DatasetProperties):

    def __init__(self, samples_per_color, **kwargs):
        super().__init__(**kwargs)
        self.samples_per_color = samples_per_color


class GenDatasetProperties(DatasetProperties):

    def __init__(self, border_matrix, **kwargs):
        super().__init__(**kwargs)
        self.border_matrix = border_matrix
