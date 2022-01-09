import random
import numpy as np
import colorsys
from torch.utils.data import Dataset
from itertools import permutations


class LuminanceDataset(Dataset):

    def __init__(self, phase, data_props, index, stimulus_generator):
        self.stim_gen = stimulus_generator
        self._prop_list = self.generateList(data_props.samples_per_class[phase])

    def __len__(self):
        return len(self._prop_list)

    def generateList(self, samples_per_class):
        prop_list = []

        for sample in range(samples_per_class):

            if random.random() > 0.3:
                grey1 = random.uniform(0, 1)
                grey2 = random.uniform(0, 1)

                class_index = int(grey1 > grey2)

                if abs(grey1-grey2) > 0.6:
                    color1 = (grey1+random.uniform(-0.15, 0.15), grey1+random.uniform(-0.15, 0.15), grey1+random.uniform(-0.15, 0.15))
                    color2 = (grey2+random.uniform(-0.15, 0.15), grey2+random.uniform(-0.15, 0.15), grey2+random.uniform(-0.15, 0.15))
                else:
                    color1 = (grey1, grey1, grey1)
                    color2 = (grey2, grey2, grey2)

                prop_list.append((class_index, color1, color2))
            else:

                if random.random() > 0.5:
                    perms = permutations([1,1,0])
                    color1 = list(perms)[0]
                    class_index = 1
                else:
                    perms = permutations([1,0,0])
                    color1 = list(perms)[0]
                    class_index = 0

                color2 = (1.0-color1[0], 1.0-color1[0], 1.0-color1[0])

                prop_list.append((class_index, color1, color2))

        return prop_list


    def __getitem__(self, idx):
        (label, color1, color2) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(color1, color2)
        return image, label



class LuminanceRangeDataset(Dataset):

    def __init__(self, range_properties, samples_per_color, stimulus_generator):
        self.stim_gen = stimulus_generator
        self._prop_list = self.generateList(range_properties, samples_per_color)

    def __len__(self):
        return len(self._prop_list)


    def generateList(self, range_property, variations_per_hue, hue_step=0.01, hue_range=(0, 1)):

        if range_property == 'cyan':
            color1 = [0.0, 0.6, 0.6]
        elif range_property == 'magenta':
            color1 = [0.6, 0.0, 0.6]
        elif range_property == 'yellow':
            color1 = [0.6, 0.6, 0.0]

        prop_list = []
        for weight in np.arange(hue_range[0], hue_range[1], hue_step):
            for sample in range(variations_per_hue):
                if range_property == 'cyan':
                    color2 = [weight, 0.0, 0.0]
                elif range_property == 'magenta':
                    color2 = [0.0, weight, 0.0]
                elif range_property == 'yellow':
                    color2 = [0.0, 0.0, weight]

                prop_list.append((int(round(weight*100)), sample, color1, color2))

        return prop_list


    def __getitem__(self, idx):
        (label, sample, color1, color2) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(color1, color2)
        return image, label, sample

