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
sys.path.insert(1, str(Path.joinpath(Path(file_dir).parent, 'generalModules')))

from figureFunctions import plotStackFor


parts = [4,5,6,7,8,9]
model_count = 150
training_bands = True
folder = 'objectClass_f40'
title_size = 24
label_size = 20
rows = 2
columns = 3
    
plt.figure(figsize=(20,12))
    
for index, part in enumerate(parts):

    plt.subplot(rows,columns,index+1)    

    plotStackFor(part, folder, training_bands=training_bands)
    
    if index < (len(parts)-3):
        plt.xticks([])
    else:
        plt.xlabel('Hue (%)', fontsize=label_size)
    
    if index % columns == 0:
        plt.ylabel('Network it.', fontsize=label_size)
    
    plt.title(f"{part} training classes", fontsize=title_size)
    
    
plt.savefig('../../data/figures/appendix/' + 'appendix_figure10.png')