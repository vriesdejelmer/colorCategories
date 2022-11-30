#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:55:31 2020

@author: vriesdejelmer
"""

import os, sys
from pathlib import Path
import matplotlib.pyplot as plt


#there has to be a nicer/cleaner way to do this
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(str(Path(file_dir)))
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent.parent, 'generalModules')))

from figureFunctions import plotStackFor, plotBorderCountFor

    #output folder
model_count = 150
layers = 6
part = 7

label_size = 24
title_size = 28
titles = ['Classification', 'Transition Counts']    
full_folder = 'objectClass_f40/'

plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=0.02, wspace=0.1)

for column in range(len((titles))):
  
    for index, layer in enumerate(range(layers)):
          
        if layer == 5:
            folder = full_folder
        else:
            folder = 'resnet18_' + str(layer) + '/'    
    
        plt.subplot(layers,len(titles),(index*len(titles))+1+column)    

        if column == 0: 
            plotStackFor(part, folder, training_bands=True)
    
            plt.yticks([])
            plt.xlim(0, 100)
        
        else:
            plotBorderCountFor(folder, 300)
            
            
        if layer == 5:
            plt.xlabel('Sample Hue (%)', fontsize=label_size)
        else:
            plt.xticks([])
                
        if column == 0:
            if layer < 5:
                plt.ylabel('Area ' + str(layer), fontsize=label_size)
            elif layer == 5:
                plt.ylabel('FC Layer', fontsize=label_size)
    
        if index == 0:
            plt.title(titles[column], fontsize=title_size)    
            
            
plt.savefig('../../data/figures/appendix/' + 'appendix_figure7.png')