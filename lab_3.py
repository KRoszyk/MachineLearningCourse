import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def plot_iris(X: np.ndarray) -> None:
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')


def ex_1():
    # Loading IRIS dataset
    iris = datasets.load_iris(as_frame=True)
    # print(iris.frame.describe())

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    # X = np.append(X, [[50,1,1,1]], axis=0)
    # y = np.append(y, [1])

    print(f'count y: {np.bincount(y)}')

    # Parametr stratify - równomierne rozłożenie danych

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, stratify=y)

    print(f'count y_train: {np.bincount(y_train)}')
    print(f'count y_test: {np.bincount(y_test)}')

    # Normalizacja danych
    # Normalizacja min-max scaler

    plot_iris(X)

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_min_max = min_max_scaler.transform(X_test)
    plot_iris(X_min_max)

    # Normalizacja Standard Scaler
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(X_train)
    X_standard = standard_scaler.transform(X_test)
    plot_iris(X_standard)

    plt.show()


def ex_2():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X = X[:, [1, 3]]    # Wybor cech ktore beda klasyfikowane, nie da sie wyswietlic wiecej cech na wykresach niz 2

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_min_max = min_max_scaler.transform(X)

    X_train = min_max_scaler.transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    clf_svm = svm.SVC(random_state=42, kernel='rbf', probability=True)
    clf_svm.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, clf_svm.predict(X_test))
    print(f'acc_svm:{acc_svm}')

    clf_linear = linear_model.LogisticRegression(random_state=42)
    clf_linear.fit(X_train, y_train)
    acc_lin = metrics.accuracy_score(y_test, clf_linear.predict(X_test))
    print(f'acc_lin:{acc_lin}')

    clf_tree = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
    clf_tree.fit(X_train, y_train)
    acc_tree = metrics.accuracy_score(y_test, clf_tree.predict(X_test))
    print(f'acc_tree:{acc_tree}')

    clf_rf = ensemble.RandomForestClassifier(random_state=42, n_estimators=1000)
    clf_rf.fit(X_train, y_train)
    acc_rf = metrics.accuracy_score(y_test, clf_rf.predict(X_test))
    print(f'acc_rf:{acc_rf}')

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    clf_gs = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, n_jobs=20, verbose=20)
    clf_gs.fit(X_train, y_train)
    print(clf_gs.cv_results_)

    # Predykcja klasy
    print(clf_svm.predict(min_max_scaler.transform([[8.0, 4.0]])))
    print(clf_svm.predict(min_max_scaler.transform([[8.0, 50.0]])))

    # Prawdopodobieństwa predykcji klas
    print(clf_svm.predict_proba(min_max_scaler.transform([[8.0, 4.0]])))
    print(clf_svm.predict_proba(min_max_scaler.transform([[8.0, 50.0]])))

    # Plotting decision regions
    plt.figure()
    plot_decision_regions(X_test, y_test, clf=clf_svm, legend=2)

    plt.figure()
    plot_decision_regions(X_test, y_test, clf=clf_linear, legend=2)

    plt.figure()
    plot_decision_regions(X_test, y_test, clf=clf_tree, legend=2)

    plt.figure()
    plot_decision_regions(X_test, y_test, clf=clf_rf, legend=2)

    plt.show()


if __name__ == '__main__':
    # ex_1()
    ex_2()