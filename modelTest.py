# **Notes**: 
# 1. RUN newPreprocessing.py BEFORE RUNNING THIS FILE TO GET THE final_data.csv FILE
# 2. I am running this with scikit-learn 1.2.2 instead of 1.3.0 because the latter had a bug with model.predict in the method knn()
    # for more information, see debugging forum: https://github.com/scikit-learn/scikit-learn/issues/26768

import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

import warnings

# Ignore FutureWarnings related to is_sparse in scikit-learn (I had to use an earlier version of sklearn bc there was a bug with KNN, so I kept getting depreciaiton warnings)
warnings.filterwarnings("ignore", category=FutureWarning)

# this gives a report on various scores to determine how well the model did:
def result_report(model, x_train, y_train, x_test, y_test, y_pred, model_name):
    perf_df=pd.DataFrame({'Train_Score': model.score(x_train, y_train), "Test_Score": model.score (x_test, y_test),
                       "Precision_Score": precision_score(y_test, y_pred, average='weighted'), "Recall_Score": recall_score(y_test, y_pred, average='weighted'),
                       "F1_Score": f1_score(y_test, y_pred, average='weighted'), "accuracy": accuracy_score(y_test, y_pred)}, index=[model_name])
    return perf_df

# these are the models tested:
def decisionTree(x_train, y_train, x_test, y_test):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    print(result_report(dt, x_train, y_train, x_test, y_test, y_pred, 'Decision Tree'))
    print('---')

def randomForest(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(result_report(model, x_train, y_train, x_test, y_test, y_pred, 'Random Forest'))
    print('---')

def knn(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    x_trainScaled = scaler.fit_transform(x_train)
    x_testScaled = scaler.transform(x_test)
    model = KNeighborsClassifier()
    model.fit(x_trainScaled, y_train)
    y_pred = model.predict(x_testScaled)
    print(result_report(model, x_trainScaled, y_train, x_testScaled, y_test, y_pred, 'K Nearest Neighbors'))
    print('---')

def gaussianNB(x_train, y_train, x_test, y_test):
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(result_report(model, x_train, y_train, x_test, y_test, y_pred, 'Naive Bayes'))
    print('---')

def main():
    """
    Main file to run from the command line.
    """

    xFeat = pd.read_csv('final_data.csv')
    
    # Assume 'Severity' is the target variable
    target_variable = 'Severity'
    
    # Split the data into features (X) and target variable (y)
    xData = xFeat.drop(columns=[target_variable])
    yData = xFeat[target_variable]

    # Use Random Under Sampling to account for majority class bias
    sam = RandomUnderSampler(random_state=0)
    xData_Resampled, yData_Resampled = sam.fit_resample(xData, yData)

    # Split the data into train and test sets (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(xData_Resampled, yData_Resampled, test_size=0.2, random_state=42)

    decisionTree(x_train, y_train, x_test, y_test)
    randomForest(x_train, y_train, x_test, y_test)
    knn(x_train, y_train, x_test, y_test)
    gaussianNB(x_train, y_train, x_test, y_test)




if __name__ == "__main__":
    main()

