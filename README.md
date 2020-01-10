# Kaggle-Titanic
[https://www.kaggle.com/c/titanic/overview](https://www.kaggle.com/c/titanic/overview)

### 0 模型准备

```
#Essentials
import pandas as pd
import numpy as np

#Plots
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

#Models
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
import xgboost
import lightgbm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from mlxtend.regressor import StackingCVRegressor
from mlxtend.classifier import StackingCVClassifier

#Misc
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import neighbors
from scipy.spec
