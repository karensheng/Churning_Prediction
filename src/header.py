'''COMMANDS'''
# %load /Users/rokushou/Desktop/header.py
# %run /Users/rokushou/Desktop/header.py
'''BASIC'''
import numpy as np
import pandas as pd
import random as rand
import math
import os
import itertools as itr
import warnings
# warnings.simplefilter('ignore')
'''PLOT'''
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
font = {'size':16}
'''STATS'''
import scipy.stats as scs
import statsmodels.api as sm
'''SCIKIT LEARN'''
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, r2_score, mean_squared_error, classification_report, make_scorer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
'''TENSORFLOW'''
import tensorflow as tf
'''PALETTE'''
t = 'setsuna'
qan = {'setsuna':'#0000CD'}
twi = '#DCB8E7' #Pale, light grayish mulberry
twi_o = '#'
twi_blu = '#273873' #Dark sapphire blue
twi_oblu = '#'
twi_pur = '#662D8A' #Moderate purple
twi_pnk = '#ED438D' #Brilliant raspberry
'''METASYNTACTIC'''
foo = bar = None
'''RETURN'''
os.system("say 'loading complete' &");
