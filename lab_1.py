from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def ex_1():

    digits = datasets.load_digits()                             # description for alt+shift+E
    # print(digits.DESCR)

    # print(digits.data[0])
    # print(digits.images[0])
    # print(digits.target[0])

    # plt.imshow(digits.images[0], cmap='gray')
    # plt.show()

    # print(digits.target_names)

    clf = svm.SVC()
    clf.fit(digits.data[0:10], digits.target[0:10])
    print(clf.predict([digits.data[-1]]))
    print([digits.target[-1]])

    pickle.dump(clf, open('clf.p', 'wb'))                       # saving and loading model with pickle
    clf_copy = pickle.load(open('./clf.p', 'rb'))


def ex_2():

    faces = datasets.fetch_olivetti_faces()
    # print(faces.DESCR)
    # print(faces.data)
    # print(faces.target)

    X, y = datasets.fetch_olivetti_faces(return_X_y=True)       # Showing data and targets
    # print(X, y)

    X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.2, shuffle=True,  random_state=42)

    figure, ax = plt.subplots(1, 5, figsize=(16, 16))

    for index, image in enumerate(X_test[0:5]):                 # Creating plot for faces from test data
        new = image.reshape(64, 64)
        ax[index].imshow(new)
        ax[index].set_title(y_test[index])
    plt.show()


def ex_3():

    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
    boston = load_boston()
    # print(X.shape)
    # print(boston['feature_names'])


def ex_4():

    x, y = datasets.make_classification(                # Testing different methods
        n_samples=100,
        n_features=3,
        n_informative=3, n_redundant=0, n_repeated=0,
        n_classes=6,
        n_clusters_per_class=1,
        class_sep=10.0,
        flip_y=0.0
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0], x[:,1], x[:,2], c=y)    # 3d plot

    # plt.scatter(x[:,0], x[:,1], x[:,2], c=y) # 2d plot

    plt.show()


def ex_5():

    d = datasets.fetch_openml(data_id=40536, as_frame=True)
    print(type(d))
    print(d['frame'])                                      # print(d.frame) data loades as dataframe


def ex_6():

    data = np.loadtxt('trainingdata.txt', delimiter=',')
    print(data)

    x = data[:, 0]
    y = data[:, 1]
    y_pred = []

    for simple_data in x:
        y_pred.append(my_regression(simple_data))

    plt.scatter(x, y)
    plt.scatter(x, y_pred, marker='*', c='r')
    plt.show()


def my_regression(x: float) -> float:
    if x >= 4:
        return 8.0
    else:
        return x * 2




def main():
    ex_6()


if __name__ == '__main__':
    main()
