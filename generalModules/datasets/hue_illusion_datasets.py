import random
import numpy as np
from .colored_word_datasets import ColorDataset

class GridDataset(ColorDataset):

    def __getitem__(self, idx):
        (label, hue, text_x_rel, text_y_rel, font_type, word) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, text_x_rel=text_x_rel, text_y_rel=text_y_rel, font_type=font_type, word=word)

        return image, label


class GridColorDataset(GridDataset):

    def __init__(self, phase, data_props, index, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(data_props.focal_hues_matrix, data_props.hue_range, data_props.samples_per_class[phase])

    def generateList(self, focal_hues, hue_range, samples):
        prop_list = []
        for index, focal_hue in enumerate(focal_hues):
            for sample in range(samples):
                hue = random.uniform(focal_hue - hue_range/2, focal_hue + hue_range/2)
                x_offset_prop = 0.5
                y_offset_prop = 0.5
                prop_list.append((index, hue, x_offset_prop, y_offset_prop))

        return prop_list

    def __getitem__(self, idx):
        (label, hue, text_x_rel, text_y_rel) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, stim_x_rel=text_x_rel, stim_y_rel=text_y_rel)

        return image, label


class GridRangeDataset(GridDataset):

    def __init__(self, range_properties, samples_per_color, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(samples_per_color)

    def generateList(self, variations_per_hue, hue_step=0.01, hue_range=(0, 1)):
        prop_list = []
        for hue in np.arange(hue_range[0], hue_range[1], hue_step):
            for sample in range(variations_per_hue):
                x_offset_prop = 0.5
                y_offset_prop = 0.5
                prop_list.append((int(round(hue*100)), sample, hue, x_offset_prop, y_offset_prop))

        return prop_list

    def __getitem__(self, idx):
        (label, sample, hue, stim_x_rel, stim_y_rel) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, stim_x_rel=stim_x_rel, stim_y_rel=stim_y_rel)

        return image, label, sample


