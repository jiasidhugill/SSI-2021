# importing packages
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import statsmodels.api as sm
import sklearn.metrics
from sklearn.metrics import r2_score

# importing data locally from computer; see README for links to datasets
house_data = pd.read_csv("house_data.csv")
sen_data = pd.read_csv("sen_data.csv")
