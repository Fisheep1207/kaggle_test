import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def preprocess():
    train_raw_data = pd.read_csv("./train.csv")
    test_raw_data = pd.read_csv("./test.csv")
    # train_raw_data = train_raw_data.sort_index(axis=1)
    # print(train_raw_data.columns)
    # print(train_raw_data.describe())
    # print(train_raw_data.head(10))
    # print(pd.unique(train_raw_data["Cabin"]))
    train_raw_data = train_raw_data.fillna(value=0)
    test_raw_data = test_raw_data.fillna(value=0)
    y_train = train_raw_data["Survived"].to_numpy()
    pass_id = test_raw_data["PassengerId"].to_numpy()
    selected_column = ["Sex", "Pclass", "Fare"]
    selected_data = train_raw_data.loc[:, selected_column]
    test_selected_data = test_raw_data.loc[:, selected_column]
    print(selected_data.describe())
    test_data = test_selected_data.to_numpy() 
    data = selected_data.to_numpy()
    data[data == "male"] = 0
    data[data == "female"] = 1
    test_data[test_data == "male"] =  0
    test_data[test_data == "female"] = 1
    data[:, 1:] = data[:, 1:] / data[:, 1:].max(axis=0)
    test_data[:, 1:] = test_data[:, 1:] / test_data[:, 1:].max(axis=0)
    # y_train = y_train.reshape(-1, 1)
    print(y_train.shape)
    return data, y_train, test_data, pass_id


if __name__ == "__main__":
    X_train, y_train, X_test, pass_id = preprocess()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=777)
    parameter_candidates = [
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
    print(X_train.shape, y_train.shape)
    clf.fit(X_train, y_train)
    print('Best score for training data:', clf.best_score_)
    print('Best `C`:',clf.best_estimator_.C)
    print('Best kernel:',clf.best_estimator_.kernel)
    print('Best `gamma`:',clf.best_estimator_.gamma)
    y_pred = clf.predict(X_valid)
    print(metrics.confusion_matrix(y_valid, y_pred))
    print(metrics.classification_report(y_valid, y_pred))
    y_test_pred = clf.predict(X_test)
    print(pass_id.shape, y_test_pred.shape)
    answer = pd.DataFrame({"PassengerId": pass_id, "Survived": y_test_pred}) 
    answer.to_csv("predict.csv", index=0)
    print(answer)