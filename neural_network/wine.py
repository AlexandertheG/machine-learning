def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

DATA_PATH = "../data"
DATA_FILE = "wine-quality.csv"

def load_data():
    csv_file_path = os.path.join(DATA_PATH, DATA_FILE)
    return pd.read_csv(csv_file_path)

def clean_data(df_set, strategy):
    imputer = Imputer(strategy=strategy)
    np_arr = imputer.fit_transform(df_set)
    return pd.DataFrame(np_arr, columns=df_set.columns)

def split_data_set(df_set):
    return train_test_split(df_set, test_size=0.2, random_state=42)

def split_stratified(df_set, lbl, test_size=0.2):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in split.split(df_set, df_set[lbl]):
        strat_train_set = df_set.loc[train_index]
        strat_test_set = df_set.loc[test_index]

    return strat_train_set, strat_test_set

def prepare_split(train_df, test_df, label_name):
    train_df = clean_data(train_df, "median")
    x_train = train_df.ix[:, train_df.columns != label_name]
    y_train = train_df.ix[:, train_df.columns == label_name]
    x_test = test_df.ix[:, test_df.columns != label_name]
    y_test = test_df.ix[:, test_df.columns == label_name]

    return x_train, y_train, x_test, y_test

def scale_fit_transform(x_train, x_test):
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test

def cross_val(train_df, labels, dec_tree_clf, k_fold=10):
    return cross_val_score(dec_tree_clf, train_df, labels.values.ravel(), cv=k_fold)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=10,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

df = load_data()

train_set, test_set = split_stratified(df, "Class")
X_train, Y_train, X_test, Y_test = prepare_split(train_set, test_set, "Class")

scaler = StandardScaler()
X_train, X_test = scale_fit_transform(X_train, X_test)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(7,), random_state=1)
clf.fit(X_train, Y_train.values.ravel())
predictions = clf.predict(X_test)
print classification_report(Y_test,predictions)

X = df.ix[:, df.columns != "Class"]
y = df.ix[:, df.columns == "Class"]

score_plt = plot_learning_curve(MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(7,), random_state=1), "MLPClassifier solver=lbfgs hidden_layer_sizes (7,)", X, y)
score_plt.show()

print "#################################################"
print "Cross validation: "
scores = cross_val(X_train, Y_train, clf)
print "10-fold cross-validation mean score: ", scores.mean()
print "10-fold cross-validation standard deviation score: ", scores.std()
print "#################################################"
