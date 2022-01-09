import random
import numpy as np
import colorsys

from .colored_word_datasets import ColorDataset

class HueColorDataset(ColorDataset):

    def __getitem__(self, idx):
        (label, hue, text_x_rel, text_y_rel, font_type, word) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, text_x_rel=text_x_rel, text_y_rel=text_y_rel, font_type=font_type, word=word)

        return image, label

class FocalColorDataset(HueColorDataset):

    def __init__(self, phase, data_props, index, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(data_props.focal_hues_matrix[index], data_props.hue_range, data_props.samples_per_class[phase])

    def generateList(self, focal_hues, hue_range, samples):
        prop_list = []
        for index, focal_hue in enumerate(focal_hues):
            for sample in range(samples):
                hue = random.uniform(focal_hue - hue_range/2, focal_hue + hue_range/2)
                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                [font_type] = random.sample(self.stim_gen.font_types, 1)
                prop_list.append((index, hue, x_offset_prop, y_offset_prop, font_type, 'Color'))

        return prop_list


class FocalMultiColorDataset(HueColorDataset):

    def __init__(self, phase, data_props, index, **kwargs):
    #def __init__(self, focal_hues, samples, hue_range, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(data_props.words, data_props.focal_hues, data_props.samples_per_class[phase])

    def generateList(self, words, focal_hues, samples):
        prop_list = []
        for index, (word, hue_range) in enumerate(zip(words, focal_hues)):
            for sample in range(samples):
                hue1 = random.uniform(hue_range[0], hue_range[1])
                hue2 = random.uniform(hue_range[0], hue_range[1])
                hue3 = random.uniform(hue_range[0], hue_range[1])
                hue4 = random.uniform(0.0, 1.0)
                hue5 = random.uniform(0.0, 1.0)
                prop_list.append((index, word, (hue1, hue2, hue3, hue4, hue5)))
        return prop_list

    def getRandomHueColor(self, brightness=0.5):
        random_color = colorsys.hsv_to_rgb(random.uniform(0.0, 1.0), 1.0, brightness)
        return (int(random_color[0]*255), int(random_color[1]*255), int(random_color[2]*255))


    def __getitem__(self, idx):
        (label, word, hues) = self._prop_list[idx]
        random_background = self.getRandomHueColor(0.5)
        image = self.stim_gen.getStimImage(hues, word=word, background_color=random_background)

        return image, label


class ColorMultiRangeDataset(HueColorDataset):

    def __init__(self, prop, samples, **kwargs):
    #def __init__(self, focal_hues, samples, hue_range, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(prop, samples)

    def generateList(self, word, samples):
        prop_list = []
        for index, hue in enumerate(np.arange(0.0,1.0, 0.01)):
            for sample in range(samples):
                hue4 = random.uniform(0.0, 1.0)
                hue5 = random.uniform(0.0, 1.0)
                prop_list.append((index, word, (hue, hue, hue, hue4, hue5), sample))
        return prop_list

    def getRandomHueColor(self, brightness=0.5):
        random_color = colorsys.hsv_to_rgb(random.uniform(0.0, 1.0), 1.0, brightness)
        return (int(random_color[0]*255), int(random_color[1]*255), int(random_color[2]*255))


    def __getitem__(self, idx):
        (label, word, hues, sample) = self._prop_list[idx]
        random_background = self.getRandomHueColor(0.5)
        image = self.stim_gen.getStimImage(hues, word=word, background_color=random_background)
        return image, label, sample


class WordRandAddBackRangeDataset(ColorDataset):

    def __init__(self, range_properties, samples_per_color, **kwargs):

        super().__init__(**kwargs)
        self._prop_list = self.generateList(range_properties, variations_per_hue=samples_per_color)

    def generateList(self, word, variations_per_hue, hue_range=(0, 1), hue_step=0.01):

        prop_list = []
        for index1, hue1 in enumerate(np.arange(hue_range[0], hue_range[1], hue_step)):
            for sample in range(variations_per_hue):
                prop_list.append((index1, word, (hue1, random.uniform(0.0, 1.0)), sample))

        return prop_list

    def __getitem__(self, idx):
        (hue_index, word_combo, hues, sample) = self._prop_list[idx]
        random_background = self.getRandomHueColor(0.5)
        image = self.stim_gen.getStimImage(hues, word_combo, random_background)
        return image, hue_index, sample

class FocalColorWordDataset(HueColorDataset):

    def __init__(self, phase, data_props, index, **kwargs):
        super().__init__(**kwargs)
        self._prop_list = self.generateList(data_props.words, data_props.focal_hues, data_props.hue_range, data_props.samples_per_class[phase])

    def generateList(self, words, focal_hues, hue_range, samples):
        prop_list = []
        for index, (word, focal_hue) in enumerate(zip(words, focal_hues)):
            for sample in range(samples):
                hue = random.uniform(focal_hue - hue_range/2, focal_hue + hue_range/2)
                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                [font_type] = random.sample(self.stim_gen.font_types, 1)
                prop_list.append((index, hue, x_offset_prop, y_offset_prop, font_type, word))

        return prop_list


class HueRangeDataset(HueColorDataset):

    def __init__(self, range_properties, samples_per_color, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(samples_per_color)

    def generateList(self, variations_per_hue, hue_step=0.01, hue_range=(0, 1)):
        prop_list = []
        for hue in np.arange(hue_range[0], hue_range[1], hue_step):
            for sample in range(variations_per_hue):
                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                [font_type] = random.sample(self.stim_gen.font_types, 1)
                prop_list.append((int(round(hue*100)), sample, hue, x_offset_prop, y_offset_prop, font_type, None))

        return prop_list

    def __getitem__(self, idx):
        (label, sample, hue, text_x_rel, text_y_rel, font_type, word) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, text_x_rel=text_x_rel, text_y_rel=text_y_rel, font_type=font_type, word=word)

        return image, label, sample


class HueWordRangeDataset(HueColorDataset):

    def __init__(self, word, samples_per_color, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(word, samples_per_color)

    def generateList(self, word, variations_per_hue, hue_step=0.01, hue_range=(0, 1)):
        prop_list = []
        for hue in np.arange(hue_range[0], hue_range[1], hue_step):
            for sample in range(variations_per_hue):
                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                [font_type] = random.sample(self.stim_gen.font_types, 1)
                prop_list.append((int(round(hue*100)), sample, hue, x_offset_prop, y_offset_prop, font_type, word))

        return prop_list

    def __getitem__(self, idx):
        (label, sample, hue, text_x_rel, text_y_rel, font_type, word) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, text_x_rel=text_x_rel, text_y_rel=text_y_rel, font_type=font_type, word=word)

        return image, label, sample


class HuePositionedRangeDataset(HueColorDataset):

    def __init__(self, range_properties, data_props, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(data_props.hue_range, data_props.hue_step, data_props.variations_per_hue)

    def generateList(self, hue_range, hue_step, variations_per_hue):
        prop_list = []
        rand_locations = np.random.random((variations_per_hue, 2))
        [font_type] = random.sample(self.stim_gen.font_types, 1)
        for hue in np.arange(hue_range[0], hue_range[1], hue_step):
            for sample in range(variations_per_hue):
                x_offset_prop, y_offset_prop = rand_locations[sample,:]

                prop_list.append((int(round(hue*100)), hue, x_offset_prop, y_offset_prop, font_type, 'Color'))

        return prop_list

    def __getitem__(self, idx):
        (label, hue, text_x_rel, text_y_rel, font_type, word) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, text_x_rel, text_y_rel, font_type, word)

        return image, label

class BoundaryHueDataset(HueColorDataset):

    #def __init__(self, phase, data_props, **kwargs):
    def __init__(self, phase, data_props, index, range_prop=0.1, **kwargs):
        super().__init__(**kwargs)
        self._prop_list = self.generateList(data_props.border_matrix[index], data_props.samples_per_class[phase], range_prop)

    def generateList(self, boundaries, samples_per_class, range_prop):
        prop_list = []
        for index in range(len(boundaries)):

            lower_border = boundaries[index]
            upper_border = boundaries[(index + 1) % len(boundaries)]

                #better to make sure lower_border lies left of upper_border
            if upper_border < lower_border:
                upper_border += 1.0

            range_width = (upper_border - lower_border)

            lower_range = (lower_border + range_width * 0.05, lower_border + range_width * (range_prop + 0.05))
            upper_range = (upper_border - range_width * (range_prop + 0.05), upper_border - range_width * 0.05)

            for sample in range(samples_per_class):

                    #we put half the samples in the lower_range and the other half in the higher range
                if sample < int(samples_per_class / 2):
                    hue = random.uniform(lower_range[0], lower_range[1])
                else:
                    hue = random.uniform(upper_range[0], upper_range[1])

                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                [font_type] = random.sample(self.stim_gen.font_types, 1)
                prop_list.append((index, hue, x_offset_prop, y_offset_prop, font_type, 'Color'))

        return prop_list


class BorderSquisherDataset(HueColorDataset):

    def __init__(self, phase, data_props, index, **kwargs):
    #def __init__(self, borders, offset, thickness, samples_per_class, **kwargs):
        super().__init__(**kwargs)
        self._prop_list = self.generateList(data_props.borders, data_props.offset, data_props.thickness, data_props.samples_per_class)

    def generateList(self, borders, offset, thickness, samples_per_class):
        prop_list = []
        for index, border in enumerate(borders):
            left_range = (border-offset, border-offset+thickness)
            right_range = (border+offset-thickness, border+offset)

            for sample in range(samples_per_class):

                if sample < int(samples_per_class / 2):
                    hue = random.uniform(left_range[0], left_range[1])
                else:
                    hue = random.uniform(right_range[0], right_range[1])

                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                [font_type] = random.sample(self.stim_gen.font_types, 1)
                prop_list.append((index, hue, x_offset_prop, y_offset_prop, font_type, 'Color'))


        for index in range(len(borders)):
            left_border = borders[index]
            right_border = borders[(index + 1) % len(borders)]

            if right_border < left_border:
                right_border += 1.0

            left_range = (left_border + offset, left_border + offset + thickness)
            right_range = (right_border - offset - thickness, right_border - offset)

            for sample in range(samples_per_class):

                if sample < int(samples_per_class / 2):
                    hue = random.uniform(left_range[0], left_range[1])
                else:
                    hue = random.uniform(right_range[0], right_range[1])

                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                [font_type] = random.sample(self.stim_gen.font_types, 1)
                prop_list.append((index + len(borders), hue, x_offset_prop, y_offset_prop, font_type, 'Color'))

        return prop_list


class WordColorDataset(HueColorDataset):

    def __init__(self, phase, data_props, index, **kwargs):
        super().__init__(**kwargs)
        self._prop_list = self.generateList(data_props.words, data_props.hue_ranges, data_props.samples_per_class[phase])

    def generateList(self, words, hue_ranges, samples):
        prop_list = []
        for index, word in enumerate(words):
            (lower_bound, upper_bound) = hue_ranges[index]
            if lower_bound > upper_bound:
                upper_bound += 1
            for sample in range(samples):
                hue = random.uniform(lower_bound, upper_bound)
                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                [font_type] = random.sample(self.stim_gen.font_types, 1)
                prop_list.append((index, hue, x_offset_prop, y_offset_prop, font_type, word))

        return prop_list

class TwoWordColorDataset(HueColorDataset):

    def __init__(self, phase, data_props, index, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(data_props.word_combos, data_props.hue_ranges, data_props.samples_per_class[phase])

    def generateList(self, word_combos, hue_combos, samples):

        prop_list = []
        for word_combo_index, (word_combo, hue_ranges) in enumerate(zip(word_combos, hue_combos)):

            for sample in range(samples):
                hue_range1 = hue_ranges[0]
                hue1 = random.uniform(hue_range1[0], hue_range1[1])
                hue_range2 = hue_ranges[1]
                hue2 = random.uniform(hue_range2[0], hue_range2[1])
                prop_list.append((word_combo_index, word_combo, (hue1, hue2)))

        return prop_list

    def __getitem__(self, idx):
        (word_combo_index, word_combo, hues) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hues, word_combo)
        return image, word_combo_index

class TwoWordHueRangeDataset(ColorDataset):

    def __init__(self, range_properties, samples_per_color, **kwargs):

        super().__init__(**kwargs)
        self._prop_list = self.generateList(range_properties, variations_per_hue=samples_per_color)

    def generateList(self, word_combo, variations_per_hue, hue_range=(0, 1), hue_step=0.02):

        prop_list = []
        for index1, hue1 in enumerate(np.arange(hue_range[0], hue_range[1], hue_step)):
            for index2, hue2 in enumerate(np.arange(hue_range[0], hue_range[1], hue_step)):
                for sample in range(variations_per_hue):
                    prop_list.append(((index1, index2), word_combo, (hue1, hue2), sample))

        return prop_list

    def __getitem__(self, idx):
        (hue_indices, word_combo, hues, sample) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hues, word_combo)
        return image, hue_indices, sample


class WordRandAdditionRangeDataset(ColorDataset):

    def __init__(self, range_properties, samples_per_color, **kwargs):

        super().__init__(**kwargs)
        self._prop_list = self.generateList(range_properties, variations_per_hue=samples_per_color)

    def generateList(self, word_combo, variations_per_hue, hue_range=(0, 1), hue_step=0.01):

        prop_list = []
        for index1, hue1 in enumerate(np.arange(hue_range[0], hue_range[1], hue_step)):
            for sample in range(variations_per_hue):
                prop_list.append((index1, word_combo, (hue1, random.uniform(0.0, 1.0)), sample))

        return prop_list

    def __getitem__(self, idx):
        (hue_index, word_combo, hues, sample) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hues, word_combo)
        return image, hue_index, sample


class TwoWordRandBackDataset(HueColorDataset):

    def __init__(self, phase, data_props, index, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(data_props.word_combos, data_props.hue_ranges, data_props.samples_per_class[phase])

    def generateList(self, word_combos, hue_combos, samples):

        prop_list = []
        for word_combo_index, (word_combo, hue_ranges) in enumerate(zip(word_combos, hue_combos)):

            for sample in range(samples):
                hue_range1 = hue_ranges[0]
                hue1 = random.uniform(hue_range1[0], hue_range1[1])
                hue_range2 = hue_ranges[1]
                hue2 = random.uniform(hue_range2[0], hue_range2[1])
                prop_list.append((word_combo_index, word_combo, (hue1, hue2)))

        return prop_list

    def getRandomHueColor(self, brightness=0.5):
        random_color = colorsys.hsv_to_rgb(random.uniform(0.0, 1.0), 1.0, brightness)
        return (int(random_color[0]*255), int(random_color[1]*255), int(random_color[2]*255))

    def __getitem__(self, idx):
        (word_combo_index, word_combo, hues) = self._prop_list[idx]
        random_background = self.getRandomHueColor(0.5)
        image = self.stim_gen.getStimImage(hues, word_combo, random_background)
        return image, word_combo_index

class WordRandAddBackRangeDataset(ColorDataset):

    def __init__(self, range_properties, samples_per_color, **kwargs):

        super().__init__(**kwargs)
        self._prop_list = self.generateList(range_properties, variations_per_hue=samples_per_color)

    def generateList(self, word_combo, variations_per_hue, hue_range=(0, 1), hue_step=0.01):

        prop_list = []
        for index1, hue1 in enumerate(np.arange(hue_range[0], hue_range[1], hue_step)):
            for sample in range(variations_per_hue):
                prop_list.append((index1, word_combo, (hue1, random.uniform(0.0, 1.0)), sample))

        return prop_list


    def getRandomHueColor(self, brightness=0.5):
        random_color = colorsys.hsv_to_rgb(random.uniform(0.0, 1.0), 1.0, brightness)
        return (int(random_color[0]*255), int(random_color[1]*255), int(random_color[2]*255))


    def __getitem__(self, idx):
        (hue_index, word_combo, hues, sample) = self._prop_list[idx]
        random_background = self.getRandomHueColor(0.5)
        image = self.stim_gen.getStimImage(hues, word_combo, random_background)
        return image, hue_index, sample

