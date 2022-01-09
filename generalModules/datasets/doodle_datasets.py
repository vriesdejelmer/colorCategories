import random
import os
import numpy as np
import re

from .colored_word_datasets import ColorDataset


class ObjectRGBColorDataset(ColorDataset):

    def __init__(self, model_phase, data_props, **kwargs):
        super().__init__(**kwargs)

        self.folder = '../data/lineDrawings/objects/'
        self._prop_list = self.generateList(model_phase, data_props.object_types, data_props.object_colors, data_props.samples_per_class[model_phase])

    def generateList(self, model_phase, object_types, object_colors, samples):
        prop_list = []
        for object_index, (object_type, object_color) in enumerate(zip(object_types, object_colors)):
            file_list = os.listdir(self.folder + object_type + '/' + model_phase)
            number_list = []
            for file_name in file_list:
                if file_name.startswith('np'):
                    reg_out = re.search(r'\d+$', file_name[:-4])
                    file_number = int(reg_out.group()) if reg_out else None
                    number_list.append(file_number)

            for sample in range(samples):
                prop_list.append((model_phase, object_index, object_type, number_list[sample], object_color))

        return prop_list

    def __getitem__(self, idx):
        (model_phase, object_index, object_type, object_num, object_color) = self._prop_list[idx]
        image = self.stim_gen.getStimRGBImage(model_phase, object_color, object_type, object_num)
        return image, object_index

class ObjectColorDataset(ColorDataset):

    def __init__(self, model_phase, data_props, index, **kwargs):
        super().__init__(**kwargs)

        self.folder = '../data/lineDrawings/objects/'
        self._prop_list = self.generateList(model_phase, data_props.object_types, data_props.object_hues, data_props.samples_per_class[model_phase])

    def generateList(self, model_phase, object_types, hue_ranges, samples):
        prop_list = []
        for object_index, (object_type, hue_range) in enumerate(zip(object_types, hue_ranges)):
            file_list = os.listdir(self.folder + object_type + '/' + model_phase)
            number_list = []
            for file_name in file_list:
                if file_name.startswith('np'):
                    reg_out = re.search(r'\d+$', file_name[:-4])
                    file_number = int(reg_out.group()) if reg_out else None
                    number_list.append(file_number)

            for sample in range(samples):
                hue = random.uniform(hue_range[0], hue_range[1])
                prop_list.append((model_phase, object_index, object_type, number_list[sample], hue))

        return prop_list

    def __getitem__(self, idx):
        (model_phase, object_index, object_type, object_num, hue) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, model_phase=model_phase, object_type=object_type, index=object_num)
        return image, object_index


class ObjectHueRangeDataset(ColorDataset):

    def __init__(self, object_type, variations_per_hue, hue_step=0.01, hue_range=(0, 1), **kwargs):
        super().__init__(**kwargs)

        self.folder = '../data/lineDrawings/objects/' + object_type + '/'
        self._prop_list = self.generateList(object_type, hue_range, hue_step, variations_per_hue)

    def generateList(self, object_type, hue_range, hue_step, variations_per_hue):
        file_list = os.listdir(self.folder + '/range')
        number_list = []
        for file_name in file_list:
            if file_name.startswith('np'):
                reg_out = re.search(r'\d+$', file_name[:-4])
                file_number = int(reg_out.group()) if reg_out else None
                number_list.append(file_number)

        prop_list = []
        for hue in np.arange(hue_range[0], hue_range[1], hue_step):
            for sample in range(variations_per_hue):
                prop_list.append((int(round(100*hue)), object_type, number_list[sample], sample, hue))

        return prop_list

    def __getitem__(self, idx):
        (object_index, object_type, object_num, sample, hue) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, model_phase='range', object_type=object_type, index=object_num)
        return image, object_index, sample

class Object2DColorDataset(ColorDataset):

    def __init__(self, model_phase, data_props, **kwargs):
        super().__init__(**kwargs)

        self.folder = '../data/lineDrawings/objects/'
        self._prop_list = self.generateList(model_phase, data_props.object_types, data_props.object_hues, data_props.samples_per_class[model_phase])

    def generateList(self, model_phase, object_combos, hue_combos, samples):
        prop_list = []
        for object_combo_index, (object_types, hue_ranges) in enumerate(zip(object_combos, hue_combos)):
            number_list1 = []
            number_list2 = []
            for object_index, object_type in enumerate(object_types):
                file_list = os.listdir(self.folder + object_type + '/' + model_phase)
                for file_name in file_list:
                    if file_name.startswith('np'):
                        reg_out = re.search(r'\d+$', file_name[:-4])
                        file_number = int(reg_out.group()) if reg_out else None
                        if object_index == 0:
                            number_list1.append(file_number)
                        else:
                            number_list2.append(file_number)


            for sample in range(samples):
                hue_range1 = hue_ranges[0]
                hue1 = random.uniform(hue_range1[0], hue_range1[1])
                hue_range2 = hue_ranges[1]
                hue2 = random.uniform(hue_range2[0], hue_range2[1])
                prop_list.append((model_phase, object_combo_index, object_types, number_list1[sample], number_list2[sample], (hue1, hue2)))

        return prop_list

    def __getitem__(self, idx):
        (model_phase, object_combo_index, object_types, object_num1, object_num2, hues) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(model_phase, hues, object_types, (object_num1, object_num2))
        return image, object_combo_index

class Object2DHueRangeDataset(ColorDataset):

    def __init__(self, object_types, variations_per_hue, hue_step=0.02, hue_range=(0, 1), **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(object_types, hue_range, hue_step, variations_per_hue)

    def getNumberList(self, object_type):
        folder = '../data/lineDrawings/objects/' + object_type + '/'

        file_list = os.listdir(folder + '/range')
        number_list = []
        for file_name in file_list:
            if file_name.startswith('np'):
                reg_out = re.search(r'\d+$', file_name[:-4])
                file_number = int(reg_out.group()) if reg_out else None
                number_list.append(file_number)

        return number_list


    def generateList(self, object_types, hue_range, hue_step, variations_per_hue):
        number_list1 = self.getNumberList(object_types[0])
        number_list2 = self.getNumberList(object_types[1])

        prop_list = []
        for index1, hue1 in enumerate(np.arange(hue_range[0], hue_range[1], hue_step)):
            for index2, hue2 in enumerate(np.arange(hue_range[0], hue_range[1], hue_step)):
                for sample in range(variations_per_hue):
                    prop_list.append(((index1, index2), object_types, (number_list1[sample], number_list2[sample]), sample, (hue1, hue2)))

        return prop_list

    def __getitem__(self, idx):
        (object_indices, object_types, object_num, sample, hues) = self._prop_list[idx]
        image = self.stim_gen.getStimImage('range', hues, object_types, object_num)
        return image, object_indices, sample


class ObjectRandomAddHueRangeDataset(Object2DHueRangeDataset):

    def generateList(self, object_types, hue_range, hue_step, variations_per_hue):
        number_list1 = self.getNumberList(object_types[0])
        number_list2 = self.getNumberList(object_types[1])

        prop_list = []
        for index1, hue1 in enumerate(np.arange(hue_range[0], hue_range[1], 0.01)):
            for sample in range(variations_per_hue):
                hue2 = random.uniform(0.0, 1.0)
                prop_list.append((index1, object_types, (number_list1[sample], number_list2[sample]), sample, (hue1, hue2)))

        return prop_list

    def __getitem__(self, idx):
        (hue_index, object_types, object_num, sample, hues) = self._prop_list[idx]
        image = self.stim_gen.getStimImage('range', hues, object_types, object_num)
        return image, hue_index, sample
