import random
import numpy as np
import colorsys

from .colored_word_datasets import ColorDataset


class ShapeColorDataset(ColorDataset):

    def __getitem__(self, idx):
        (label, hue, stim_x_rel, stim_y_rel) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, stim_x_rel=stim_x_rel, stim_y_rel=stim_y_rel)

        return image, label


class FocalColorDataset(ShapeColorDataset):

    def __init__(self, phase, data_props, index, **kwargs):
    #def __init__(self, focal_hues, samples, hue_range=0.02, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(data_props.focal_hues_matrix[index], data_props.hue_range, data_props.samples_per_class[phase])

    def generateList(self, focal_hues, hue_range, samples):
        prop_list = []
        for index, focal_hue in enumerate(focal_hues):
            for sample in range(samples):
                hue = random.uniform(focal_hue - hue_range/2, focal_hue + hue_range/2)
                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                prop_list.append((index, hue, x_offset_prop, y_offset_prop))

        return prop_list



class HueRangeDataset(ShapeColorDataset):

    def __init__(self, range_properties, samples_per_color, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(samples_per_color)

    def generateList(self, variations_per_hue, hue_step=0.01, hue_range=(0, 1)):
        prop_list = []
        for hue in np.arange(hue_range[0], hue_range[1], hue_step):
            for sample in range(variations_per_hue):
                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                prop_list.append((int(round(hue*100)), hue, x_offset_prop, y_offset_prop, sample))
        return prop_list


    def __getitem__(self, idx):
        (label, hue, stim_x_rel, stim_y_rel, sample) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, stim_x_rel=stim_x_rel, stim_y_rel=stim_y_rel)

        return image, label, sample