def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
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

def cross_val(train_df, labels, dec_tree_clf, k_fold=10):
    return cross_val_score(dec_tree_clf, train_df, labels.values.ravel(), cv=k_fold)

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

X = df.ix[:, df.columns != "Class"]
y = df.ix[:, df.columns == "Class"]

train_set, test_set = split_stratified(df, "Class")
X_train, Y_train, X_test, Y_test = prepare_split(train_set, test_set, "Class")

err_val_arr = []
n_estim_arr = []

for n_estim in np.arange(50, 300, 5):
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=n_estim,
                                                    learning_rate=0.1, random_state=0)
    clf.fit(X_train, Y_train.values.ravel())
    scr = clf.score(X_test, Y_test)
    err_val_arr.append(1 - scr)
    n_estim_arr.append(n_estim)

plt.plot(n_estim_arr, err_val_arr, alpha=0.5)
plt.ylabel('err')
plt.xlabel('n_estimators w/ max_depth=5 learning_rate=0.1')
plt.show()


clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=240,
                                                learning_rate=0.1, random_state=0)
print "#################################################"
print "10-fold cross-validation of AdaBoostClassifier w/"
print "DecisionTreeClassifier(max_depth=5) as base_estimator and n_estimators=240:\n"
scores = cross_val(X_train, Y_train, clf)
print "Mean score: ", scores.mean()
print "Standard deviation score: ", scores.std()
print "#################################################"
print "Train AdaBoostClassifier model..."
clf.fit(X_train, Y_train.values.ravel())
print "Accuracy of prediction:", clf.score(X_test, Y_test)
print "#################################################\n"

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=240,
                                                    learning_rate=0.1, random_state=0)
score_plt = plot_learning_curve(clf, "AdaBoostClassifier max_depth=5 n_estimators=240 learning_rate=0.1", X, y)
score_plt.show()

err_val_arr = []
n_estim_arr = []

for n_estim in np.arange(50, 300, 5):
    clf = GradientBoostingClassifier(n_estimators=n_estim, learning_rate=0.1,
                                     max_depth=5, random_state=0)
    clf.fit(X_train, Y_train.values.ravel())
    scr = clf.score(X_test, Y_test)
    err_val_arr.append(1 - scr)
    n_estim_arr.append(n_estim)

plt.plot(n_estim_arr, err_val_arr, alpha=0.5)
plt.ylabel('err')
plt.xlabel('n_estimators w/ max_depth=5 learning_rate=0.1')
plt.show()

clf = GradientBoostingClassifier(n_estimators=190, learning_rate=0.1,
                                 max_depth=5, random_state=0)
print "#################################################"
print "10-fold cross-validation GradientBoostingClassifier w/ n_estimators=190 max_depth=5: "
scores = cross_val(X_train, Y_train, clf)
print "Mean score: ", scores.mean()
print "Standard deviation score: ", scores.std()
print "#################################################"
print "Train GradientBoostingClassifier model..."
clf.fit(X_train, Y_train.values.ravel())
print "Accuracy of prediction:", clf.score(X_test, Y_test)
print "#################################################\n"

# clf = GradientBoostingClassifier(n_estimators=190, learning_rate=0.1,
#                                  max_depth=5, random_state=0)
# score_plt = plot_learning_curve(clf, "GradientBoostingClassifier max_depth=5 n_estimators=190 learning_rate=0.1", X, y)
# score_plt.show()
