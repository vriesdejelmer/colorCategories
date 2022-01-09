from torch.utils.data import Dataset
import random, string
import numpy as np

class ColorDataset(Dataset):

    def __init__(self, stimulus_generator):
        self.stim_gen = stimulus_generator

    def __len__(self):
        return len(self._prop_list)

    def getRandomColor(self, distance_color=None, min_distance=50):
        random_color = np.random.random(3) * 255
        if not distance_color is None:
            while np.linalg.norm(random_color-distance_color) < min_distance:
                random_color = np.random.random(3) * 255
        return (int(random_color[0]), int(random_color[1]), int(random_color[2]))


class WordRandColorDataset(ColorDataset):

    def __init__(self, phase, data_props, index, **kwargs):
        super().__init__(**kwargs)

        self._prop_list = self.generateList(data_props.words, data_props.samples_per_class[phase])
    
    
    def generateList(self, words, samples_per_class):
        prop_list = []
        for class_index, word in enumerate(words):

            for sample in range(samples_per_class):
                x_offset_prop = random.uniform(0, 1)
                y_offset_prop = random.uniform(0, 1)
                [font_type] = random.sample(self.stim_gen.font_types, 1)

                hue = random.uniform(0, 1)
                
                prop_list.append((class_index, hue, x_offset_prop, y_offset_prop, font_type, word))

        return prop_list


    def __getitem__(self, idx):
        (label, hue, text_x_rel, text_y_rel, font_type, word) = self._prop_list[idx]
        image = self.stim_gen.getStimImage(hue, text_x_rel=text_x_rel, text_y_rel=text_y_rel, font_type=font_type, word=word)
        
        return image, label


# class ColoredWordGenerator(ColorDataset):

#     def __init__(self, colors, color_names, samples_per_class, **kwargs):
#         super().__init__(colors, **kwargs)

#         self.colors = colors
#         self.prop_list = self.generateList(color_names, samples_per_class)

#     def generateList(self, color_names, samples_per_class):
#         prop_list = []
#         for index in range(len(self.colors)):

#             for sample in range(samples_per_class):
#                 x_offset_prop = random.uniform(0, 1)
#                 y_offset_prop = random.uniform(0, 1)
#                 [font_type] = random.sample(self.font_types, 1)

#                 colorList = list(range(len(self.colors)))
#                 colorList.remove(index)
#                 [bg_index] = random.sample(colorList, 1)

#                 prop_list.append((index, bg_index, x_offset_prop, y_offset_prop, font_type, color_names[index]))

#         return prop_list

#     def getRandomString(self, length):
#         letters = string.ascii_lowercase
#         result_str = ''.join(random.choice(letters) for i in range(length))
#         return result_str.capitalize()


#     def getColorFromList(self, index):
#         return self.colors[index]


#     def getImage(self, idx):
#         (label, bg_index, text_x_rel, text_y_rel, font_type, word) = self.prop_list[idx]
#         foreground_color = self.getColorFromList(label)
#         image = self.stim_gen.getStimImage(foreground_color, self.stim_gen.background_color, text_x_rel, text_y_rel, font_type, word)
#         return image, label


#     def __getitem__(self, idx):
#         image, label = self.getImage(idx)
#         # generate image here

#         return image, label


# class RGBRangeDataset(ColorDataset):

#     def __init__(self, variations_per_hue, steps=25, **kwargs):
#         super().__init__(**kwargs)

#         self.prop_list = self.generateList(steps, variations_per_hue)

#     def generateList(self, steps, variations_per_hue):
#         prop_list = []
#         for index_r, red in enumerate(np.linspace(0, 255, steps)):
#             for index_g, green in enumerate(np.linspace(0, 255, steps)):
#                 for index_b, blue in enumerate(np.linspace(0, 255, steps)):
#                     for sample in range(variations_per_hue):
#                         x_offset_prop = random.uniform(0, 1)
#                         y_offset_prop = random.uniform(0, 1)
#                         [font_type] = random.sample(self.stim_gen.font_types, 1)
#                         prop_list.append((index_r, index_g, index_b, sample, int(red), int(green), int(blue), x_offset_prop, y_offset_prop, font_type, 'Color'))
#         return prop_list

#     def __getitem__(self, idx):
#         (index_r, index_g, index_b, sample, red, green, blue, text_x_rel, text_y_rel, font_type, word) = self.prop_list[idx]
#         image = self.stim_gen.getStimImage((red, green, blue), text_x_rel, text_y_rel, font_type, word)

#         return image, (index_r, index_g, index_b), sample


# class RGBColorDifferenceDataset(ColorDataset):

#     def __init__(self, samples, variations_per_location, **kwargs):
#         super().__init__(**kwargs)

#         self.prop_list = self.generateList(samples, variations_per_location)

#     def generateList(self, samples, variations_per_location):
#         prop_list = []

#         for sample_index in range(samples):
#             x_offset_prop = random.uniform(0, 1)
#             y_offset_prop = random.uniform(0, 1)
#             [font_type] = random.sample(self.stim_gen.font_types, 1)
#             for color_index in range(variations_per_location):
#                 (red, green, blue) = self.getRandomColor()
#                 prop_list.append((color_index, int(red), int(green), int(blue), x_offset_prop, y_offset_prop, font_type, 'Color'))
#         return prop_list

#     def __getitem__(self, idx):
#         (label, red, green, blue, text_x_rel, text_y_rel, font_type, word) = self.prop_list[idx]
#         image = self.stim_gen.getStimImage((red, green, blue), text_x_rel, text_y_rel, font_type, word)

#         return image, label


# class RGBLocationDifferenceDataset(ColorDataset):

#     def __init__(self, samples, variations_per_color, **kwargs):
#         super().__init__(**kwargs)

#         self.prop_list = self.generateList(samples, variations_per_color)

#     def generateList(self, samples, variations_per_color):
#         prop_list = []

#         for sample_index in range(samples):
#             (red, green, blue) = self.getRandomColor()
#             for location_index in range(variations_per_color):
#                 x_offset_prop = random.uniform(0, 1)
#                 y_offset_prop = random.uniform(0, 1)
#                 [font_type] = random.sample(self.stim_gen.font_types, 1)
#                 prop_list.append((location_index, int(red), int(green), int(blue), x_offset_prop, y_offset_prop, font_type, 'Color'))
#         return prop_list

#     def __getitem__(self, idx):
#         (label, red, green, blue, text_x_rel, text_y_rel, font_type, word) = self.prop_list[idx]
#         image = self.stim_gen.getStimImage((red, green, blue), text_x_rel, text_y_rel, font_type, word)

#         return image, label


# class RGBDiscriminationDataset(ColorDataset):

#     def __init__(self, samples, **kwargs):
#         super().__init__(**kwargs)

#         self.prop_list = self.generateList(samples)

#     def generateList(self, samples):
#         prop_list = []

#         for sample_index in range(samples):
#             for type_index in range(2):

#                 (red1, green1, blue1) = self.getRandomColor()
#                 x_offset_prop1 = random.uniform(0, 1)
#                 y_offset_prop1 = random.uniform(0, 1)
#                 [font_type1] = random.sample(self.stim_gen.font_types, 1)
#                 stim_description1 = StimDescription((red1, green1, blue1), (x_offset_prop1, y_offset_prop1), font_type1, 'Color')

#                 if type_index == 0:
#                     (red2, green2, blue2) = (red1, green1, blue1)
#                 else:
#                     (red2, green2, blue2) = self.getRandomColor(distance_color=(red1, green1, blue1), min_distance=50)
#                 x_offset_prop2 = random.uniform(0, 1)
#                 y_offset_prop2 = random.uniform(0, 1)
#                 [font_type2] = random.sample(self.stim_gen.font_types, 1)
#                 stim_description2 = StimDescription((red2, green2, blue2),(x_offset_prop2, y_offset_prop2), font_type2, 'Color')

#                 prop_list.append((stim_description1, stim_description2, type_index))
#         return prop_list

#     def __getitem__(self, idx):
#         (stim_description1, stim_description2, label) = self.prop_list[idx]
#         image1 = self.stim_gen.getStimImage(stim_description1.foreground_color, stim_description1.text_location[0], stim_description1.text_location[1], stim_description1.font_type, stim_description1.word)
#         image2 = self.stim_gen.getStimImage(stim_description2.foreground_color, stim_description2.text_location[0], stim_description2.text_location[1], stim_description2.font_type, stim_description2.word)

#         return image1, image2, label


# class RGBMultDiscriminationDataset(ColorDataset):

#     def __init__(self, samples, samples_per_color, **kwargs):
#         super().__init__(**kwargs)

#         self.prop_list = self.generateList(samples, samples_per_color)

#     def generateList(self, samples, samples_per_color):
#         prop_list = []

#         for sample_index in range(samples):
#             (red1, green1, blue1) = self.getRandomColor()
#             (red2, green2, blue2) = self.getRandomColor(distance_color=(red1, green1, blue1), min_distance=30)

#             for _ in range(samples_per_color):

#                 x_offset_prop1 = random.uniform(0, 1)
#                 y_offset_prop1 = random.uniform(0, 1)
#                 [font_type] = random.sample(self.stim_gen.font_types, 1)
#                 stim_description1 = StimDescription(self.jitterColor((red1,green1,blue1), 1), (x_offset_prop1, y_offset_prop1), font_type, 'Color')
#                 stim_description2 = StimDescription(self.jitterColor((red2,green2,blue2), 1), (x_offset_prop1, y_offset_prop1), font_type, 'Color')

#                 prop_list.append((stim_description1, stim_description2))
#         return prop_list


#     def jitterColor(self, color, sd):
#         return (color[0]+int(np.random.randn()*sd), color[1]+int(np.random.randn()*sd), color[2]+int(np.random.randn()*sd))


#     def __getitem__(self, idx):
#         (stim_description1, stim_description2) = self.prop_list[idx]
#         image1 = self.stim_gen.getStimImage(stim_description1.foreground_color, stim_description1.text_location[0], stim_description1.text_location[1], stim_description1.font_type, stim_description1.word)
#         image2 = self.stim_gen.getStimImage(stim_description2.foreground_color, stim_description2.text_location[0], stim_description2.text_location[1], stim_description2.font_type, stim_description2.word)

#         return image1, image2


# class StimDescription:

#     def __init__(self, foreground_color, text_location, font_type, word):
#         self.foreground_color = foreground_color
#         self.text_location = text_location
#         self.font_type = font_type
#         self.word = word



# class RandomWordGenerator(ColoredWordGenerator):

#     def __init__(self, colors, samples_per_class, transform, image_properties):
#         super().__init__(colors, transform, image_properties)
#         self.prop_list = self.generateList(samples_per_class)

#     def generateList(self, colors, samples_per_class):
#         prop_list = []
#         for index in range(len(colors)):

#             for sample in range(samples_per_class):
#                 x_offset_prop = random.uniform(0, 1)
#                 y_offset_prop = random.uniform(0, 1)
#                 [font_type] = random.sample(self.font_types, 1)

#                 colorList = list(range(len(colors)))
#                 colorList.remove(index)
#                 [bg_index] = random.sample(colorList, 1)

#                 prop_list.append((index, bg_index, x_offset_prop, y_offset_prop, font_type, self.getRandomString(5)))

#         return prop_list


# class RandomColorGenerator(ColoredWordGenerator):

#     def __init__(self, samples_per_class, transform, image_properties):
#         super().__init__(colors, transform, image_properties)

#         self.transform = transform

#         self.prop_list = self.generateList(color_names, samples_per_class)

#     def generateList(self, color_names, samples_per_class):

#         prop_list = []

#         for index in range(len(color_names)):

#             for sample in range(samples_per_class):

#                 x_offset_prop = random.uniform(0, 1)
#                 y_offset_prop = random.uniform(0, 1)
#                 [font_type] = random.sample(self.font_types, 1)

#                 prop_list.append((index, x_offset_prop, y_offset_prop, font_type, color_names[index]))

#         return prop_list

#     def getRandomColor(self, distance_color=None, min_distance=50):
#         random_color = np.random.random(3) * 255
#         if not distance_color is None:
#             while np.linalg.norm(random_color-distance_color) < min_distance:
#                 random_color = np.random.random(3) * 255
#         return (int(random_color[0]), int(random_color[1]), int(random_color[2]))

#     def getImage(self, idx):
#         (label, text_x_rel, text_y_rel, font_type, word) = self.prop_list[idx]
#         foreground_color = self.getRandomColor(distance_color=self.image_props.background_color)
#         image = self.getWordImage(foreground_color, self.image_props.background_color, text_x_rel, text_y_rel, font_type, word)
#         return image, label


#     def __getitem__(self, idx):
#         image, label = self.getImage(idx)
#         # generate image here
#         if not self.transform is None:
#             image = self.transform(image)

#         return image, label

