# Breast-Cancer
This is a comparison of available binary classifiers in Scikit-Learn on the breast cancer (BC) dataset. The BC dataset comes with the Scikit-Learn package itself.
Scope: QC comparison of Scikit-Learn binary classifiers - Logistic Regression (LR), Extra Trees (ET), Gradient Boosting (GB), Random Forest (RF), and KNN.
Content: reading input data, importing key libraries, Exploratory Data Analysis (EDA), ML data preparation (scaling and test/train/target/features data splitting), ML model training, test data predictions, and classification QC analysis using available metrics and Scikit-Plot curves.
Cumulative gains and lift curves (ML performance) are identical for all classifiers
PCA: Silhouette score 0.116, number of clusters is 3 (elbow plot), 0.982 explained variance ratio for first 1 components.
ML classification report: ROC AUC score 0.98 (KNN), F1-score 0.97 (LR), FP=3% (LR).
PCA 2D projection: good separation of classes 0 and 1.
#breastcancer #BreastCancerAwarenessMonth #breastcancerawareness #womenhealth #CancerAwarenessMonth #machinelearning #artificialintelligence #datascience #kaggle #python #scikitlearn #datavisualization #classifications #model #Training #testing #validation #confidence #diagnosis #diagnostics #predictions #healthcare #health #publichealth #screening
Explore More:
https://wp.me/pdMwZd-2SX
Youtube link:
https://youtu.be/BRN3W_HR3FA

import scikitplot as skplt

import sklearn
from sklearn.datasets import load_digits, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import sys
import warnings
warnings.filterwarnings("ignore")

print("Scikit Plot Version : ", skplt.__version__)
print("Scikit Learn Version : ", sklearn.__version__)
print("Python Version : ", sys.version)

%matplotlib inline

Scikit Plot Version :  0.3.7
Scikit Learn Version :  1.1.3
Python Version :  3.9.13 (main, Aug 25 2022, 23:51:50)
