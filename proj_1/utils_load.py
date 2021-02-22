
import os
import numpy as np
import pandas as pd
import pickle as pkl


def load_planar_data():
    if os.path.exists('data/planar_data.pkl'):
        x, y = pkl.load(open('data/planar_data.pkl', 'rb'))
    return x, y


def load_phish_data():
    if os.path.exists('data/phish_data.pkl'):
        x, y = pkl.load(open('data/phish_data.pkl', 'rb'))
    return x, y

