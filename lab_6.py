import missingno as msno
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import impute
from sklearn import ensemble
from sklearn import svm
import seaborn as sns


def ex_1():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    random.seed(42)

    X, y = datasets.fetch_openml(name='Titanic', version=1, return_X_y=True, as_frame=True)
    X: pd.DataFrame = X
    y: pd.DataFrame = y

    # print(X.head(5))
    # print(X.info())
    # print(X.describe())

    X.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)
    # print(X.head(5))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    # print(X_train.info())

    y_predicted_random = random.choices(['0', '1'], k=len(y_test))
    print(y_predicted_random[0:5])

    print(metrics.classification_report(y_test, y_predicted_random))

    y_predict_0 = ['0']*len(y_test)
    # print(metrics.classification_report(y_test, y_predict_0))

    # print(f'y.value_counts():\n{y.value_counts()}')

    # msno.matrix(X)
    # msno.heatmap(X)

    X_combined = pd.concat([X_train, y_train.astype(float)], axis=1)
    print(X_combined.head(5))

    df_temp = X_combined[['sex', 'survived']].groupby('sex').mean()
    print(df_temp.head(5))

    df_temp = X_combined[['pclass', 'survived']].groupby('pclass').mean()
    print(df_temp.head(5))

    X_combined['sex'].replace({'male':0, 'female':1}, inplace=True)
    print(X_combined.corr())

    X_combined.boxplot()

    # g = sns.heatmap(X_combined.corr(), annot=True, cmap='coolwarm')
    plt.show()


if __name__ == '__main__':
    ex_1()

