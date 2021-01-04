import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import svm
from sklearn import impute
from sklearn import ensemble
from sklearn.experimental import enable_iterative_imputer
import seaborn as sns


def plot_iris2d(X: np.ndarray, y: np.ndarray) -> None:
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)


def ex_1():
    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)
    # print(X)

    # print(X.info())
    # print(X.describe())

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_2 = X_train.copy()

    plt.figure()
    X_train.boxplot()
    X_train.hist(bins=20)
    plt.figure()
    sns.boxplot(x=X_train['mass'])

    imputer_mass = impute.SimpleImputer(missing_values=0.0, strategy='mean')
    imputer_skin = impute.SimpleImputer(missing_values=0.0, strategy='mean')

    X_train[['mass']] = imputer_mass.fit_transform(X_train[['mass']])
    X_train[['skin']] = imputer_skin.fit_transform(X_train[['skin']])

    X_test[['mass']] = imputer_mass.transform(X_test[['mass']])
    X_test[['skin']] = imputer_mass.transform(X_test[['skin']])

    df_mass = X_train[['mass']]
    # print(df_mass.head(5))

    # Wykrywanie anomalii czyli odstających danych

    X_train_isolation = X_train.values
    X_train_isolation = X_train_isolation[:, [1, 5]]
    X_test_isolation = X_test.values
    X_test_isolation = X_test_isolation[:, [1, 5]]

    isolation_forest = ensemble.IsolationForest(contamination=0.05)
    isolation_forest.fit(X_train_isolation)
    y_predicted_outliers = isolation_forest.predict(X_test_isolation)
    print(y_predicted_outliers)

    plot_iris2d(X_test_isolation, y_predicted_outliers)

    clf = svm.SVC(random_state=42)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted))

    X_train.hist()

    imputer_it = impute.IterativeImputer(missing_values=0.0)

    X_train_2[['mass']] = imputer_it.fit_transform(X_train_2[['mass']])
    X_train_2[['skin']] = imputer_it.fit_transform(X_train_2[['skin']])

    X_train_2.hist(bins=20)
    plt.figure()
    X_train_2.boxplot()

    clf_rf = ensemble.RandomForestClassifier(random_state=42)
    clf_rf.fit(X_train, y_train)
    y_predicted = clf_rf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted))

    importances = clf_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


if __name__ == '__main__':
    ex_1()