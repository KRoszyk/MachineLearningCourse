import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn import pipeline, cluster
from mpl_toolkits.mplot3d import Axes3D
import gap_statistic
from sklearn.decomposition import PCA
from sklearn import manifold


def plot_iris2d(X: np.ndarray, y: np.ndarray) -> None:
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.axvline(x=0)                    # Osie x i y
    plt.axhline(y=0)


def plot_iris(X: np.ndarray, y: np.ndarray) -> None:
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 2], c=y)
    # plt.axvline(x=0)                    # Osie x i y
    # plt.axhline(y=0)
    # plt.title('Iris sepal features')
    # plt.xlabel('sepal length (cm)')
    # plt.ylabel('sepal width (cm)')

    # Wykres 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)


def ex_1():
    # Loading IRIS dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Metody klasteryzacji
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(X)
    kmeans_3 = kmeans.labels_
    # print(kmeans.labels_)
    # print(y)

    kmeans = cluster.KMeans()
    kmeans.fit(X)
    kmeans_default = kmeans.predict(X)

    db = cluster.DBSCAN().fit(X)
    db_labels = db.labels_

    clustering = cluster.SpectralClustering(n_clusters=8).fit(X)
    spectr_labels = clustering.labels_

    affinity = cluster.AffinityPropagation(random_state=43).fit(X)
    affinity_labels = affinity.labels_
    # print(np.unique(affinity_labels))

    print('adjusted_rand_score:')
    print(f'kmeans_3: {metrics.adjusted_rand_score(y, kmeans_3)}')
    print(f'kmeans_default: {metrics.adjusted_rand_score(y, kmeans_default)}')
    print(f'db_labels: {metrics.adjusted_rand_score(y, db_labels)}')
    print(f'spectr_labels: {metrics.adjusted_rand_score(y, spectr_labels)}')
    print(f'affinity_labels: {metrics.adjusted_rand_score(y, affinity_labels)}')

    print('\ncalinski_harabasz_score:')
    print(f'kmeans_3: {metrics.calinski_harabasz_score(X, kmeans_3)}')
    print(f'kmeans_default: {metrics.calinski_harabasz_score(X, kmeans_default)}')
    print(f'db_labels: {metrics.calinski_harabasz_score(X, db_labels)}')
    print(f'spectr_labels: {metrics.calinski_harabasz_score(X, spectr_labels)}')
    print(f'affinity_labels: {metrics.calinski_harabasz_score(X, affinity_labels)}')

    # plot_iris(X, y)
    # plot_iris(X, kmeans_3)
    # plot_iris(X, kmeans_default)
    # plot_iris(X, db_labels)
    # plot_iris(X, spectr_labels)
    # plot_iris(X, affinity_labels)
    # plt.show()


def ex_2():
    # Loading IRIS dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Metoda Gap_statistic
    optimalK = gap_statistic.OptimalK(n_jobs=4, parallel_backend='joblib')
    n_clusters = optimalK(X, cluster_array=range(2, 15))
    print(optimalK.gap_df.head())
    optimalK.plot_results()

    # Metoda łokcia
    clusters = range(2, 15)
    inertias = []
    for n in clusters:
        kmeans = cluster.KMeans(n_clusters=n).fit(X)
        inertias.append(kmeans.inertia_)

    plt.plot(clusters, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(clusters)
    plt.show()


def ex_3():
    # Separacja danych za pomocą PCA i TSNE
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    tsne = manifold.TSNE(n_components=2)
    tsne.fit(X)
    X_tsne = tsne.fit_transform(X)

    plot_iris2d(X, y)
    plot_iris2d(X_pca, y)
    plot_iris2d(X_tsne, y)
    plt.show()


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    ex_3()
