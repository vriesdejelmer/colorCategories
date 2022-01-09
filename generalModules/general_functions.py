#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 21:32:00 2021

@author: vriesdejelmer
"""

import csv
import torch

import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.pyplot as plt

def writeProperties(new_vars, old_vars, data_dir):

    with open(data_dir + 'properties.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key in set(new_vars) - set(old_vars):
            if key != 'old_vars':
                writer.writerow([key, new_vars[key]])


def getDevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if not (str(device) == "cuda"):
        print("WE'RE NOT RUNNING ON CUDA")

    return device

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=5, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


    #convert cartesian coordinates to polar
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

    #convert proportions to cartesian coords on the unit circle
def prop2unit(prop):
    x_coord = np.cos(prop*2*np.pi)
    y_coord = np.sin(prop*2*np.pi)
    return(x_coord, y_coord)

