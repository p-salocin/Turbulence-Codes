# Created Date: Wednesday, January 7th 2026, 3:48:06 pm
# Author: Iván R. R. Gonzáles
# Editor: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Importing necessary libraries


import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import igraph as ig
import umap
from sklearn.mixture import GaussianMixture

# Load the file .mat and extract data

file_path = r'C:\Users\nicol\Documents\Turbulence Codes\Data\Bet286.mat' # Insert the path to your .mat file here
mat_contents = scipy.io.loadmat(file_path)
data = mat_contents['Bet286']



