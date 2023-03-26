#!/usr/bin/env python3
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn import metrics, preprocessing
import numpy as np
import pandas as pd
import glob
import os

from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import f1_score

import glob
import os
import gzip
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC




def load_csv(filename):
  df = pd.read_csv(filename)
  # features=df.iloc[:,0:-2]
  # filenames = df.iloc[:,-2]
  labels=df.iloc[:,-1]
  return df, labels

def remove_classes(dataframe, lbl):
    # find removable classes if they have less than 10 instances
    removables = [label for label, count in Counter(lbl).items() if count <= 10]
    print(removables)
    dataframe = dataframe[~dataframe[dataframe.columns[-1]].isin(removables)]
    features=dataframe.iloc[:,0:-2]
    # filenames = dataframe.iloc[:,-2]
    labels=dataframe.iloc[:,-1]
    return features, labels



if __name__ == "__main__":
    # i x vectors
    #df, labels = load_csv("cv_ivectors.csv")
    #features, labels = remove_classes(df, labels)
    #print(len(labels)), print(len(features))
    #X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42,shuffle=True)

    ## mfcc vector
    df, labels = load_csv("archimob_train_mfcc_vectors.csv")
    X_train, y_train = remove_classes(df, labels)
    print(len(X_train)), print(len(y_train))
    df, labels = load_csv("archimob_test_mfcc_vectors.csv")
    X_test, y_test = remove_classes(df, labels)
    print(len(X_test)), print(len(y_test))




    # small test
    lsvc = LinearSVC(verbose=0)
    print(lsvc)
    LinearSVC(C=1.0)
    lsvc.fit(X_train, y_train)
    score = lsvc.score(X_train, y_train)
    print("Score: ", score)

    y_pred = lsvc.predict(X_test)
    print("Accuracy on Linear SVC :",accuracy_score(y_test, y_pred))
    print(f"Report Classifier: vectors/SVC Linear ")
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    print("F1:", f1_score(y_test, y_pred, average='macro'))

    cf_matrix = confusion_matrix(y_test, y_pred)
    svm_plot = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
                fmt='.2%', cmap='Blues')
    figure = svm_plot.get_figure()
    figure.savefig('small_svm_conf_percentage.png', dpi=400)

    svm_plot = sns.heatmap(cf_matrix, annot=True,
                fmt='.2%', cmap='Blues')
    figure = svm_plot.get_figure()
    figure.savefig('small_svm_conf.png', dpi=400)





    # # Gridsearch to determine the value of C
    param_grid = {'C':np.arange(1, 100, 10)}
    linearSVC = GridSearchCV(LinearSVC(),param_grid,cv=5,return_train_score=True)
    linearSVC.fit(X_train,y_train)
    print(linearSVC.best_params_)

    bestlinearSVC = linearSVC.best_estimator_
    bestlinearSVC.fit(X_train,y_train)
    bestlinearSVC.score(X_train,y_train)
    y_pred = bestlinearSVC.predict(X_test)
    print("Accuracy on Linear SVC :",accuracy_score(y_test, y_pred))
    print(f"Report Classifier: vectors/SVC Linear ")
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    print("F1:", f1_score(y_test, y_pred, average='macro'))

    cf_matrix = confusion_matrix(y_test, y_pred)
    svm_plot = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
                fmt='.2%', cmap='Blues')
    figure = svm_plot.get_figure()
    figure.savefig('isvm_conf_percentage.png', dpi=400)

    svm_plot = sns.heatmap(cf_matrix, annot=True,
                fmt='.2%', cmap='Blues')
    figure = svm_plot.get_figure()
    figure.savefig('isvm_conf.png', dpi=400)





