from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score, v_measure_score
from base_class import *
from sklearn.cluster import KMeans
from math import sqrt
from random import randint
import matplotlib.pyplot as plt
import numpy as np


class Clustering:
    def __init__(self, features=[], max_iter=100, n_clusters=3, n_init=10, init_type='random'):
        self.features: List[List[float]] = features  # list of features for clustering
        self.max_iter: int = max_iter  # Maximum number of iterations of the k-means algorithm for a single run.
        self.clustering_method = None
        self.n_clusters: int = n_clusters  # The number of clusters to form as well as the number of centroids to generate
        self.n_init: int = n_init  # Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
        self.labels: List[int] = []
        self.init_type: str = init_type

    def run(self, sklearn_method=True, show_logs=True):
        """
        Cluster features data.
        :param sklearn_method(bool):    if True then sklearn KMeans class fit() method is used. If False then authors method is used.
        :return:
        """
        if sklearn_method:
            self.clustering_method = KMeans(n_clusters=self.n_clusters, random_state=0, max_iter=self.max_iter,
                                            init=self.init_type, n_init=self.n_init).fit(
                self.features)
            self.labels = self.clustering_method.labels_
            print("kmeans.labels_", self.clustering_method.labels_)
            print("kmeans.n_iter_", self.clustering_method.n_iter_)
            print("kmeans.inertia_", self.clustering_method.inertia_)
        else:
            self.clustering_method = self.fit(show_logs=show_logs)
            self.labels = self.clustering_method
            print("labels: ", self.labels)

    def initialize_centroids(self, init_type):
        centroids = {}
        if init_type == 'random':
            selected_centroids = []
            for i in range(self.n_clusters):
                rand_fs_idx = randint(0, len(self.features) - 1)  # random feature set index
                while rand_fs_idx in selected_centroids:
                    rand_fs_idx = randint(0, len(self.features) - 1)
                centroids[i] = self.features[rand_fs_idx]
                selected_centroids.append(rand_fs_idx)
        elif init_type == 'k-means++':
            '''Initialize one point at random.
                loop for k - 1 iterations:
                    Next, calculate for each point the distance of the point from its nearest center. Sample a point with a 
                    probability proportional to the square of the distance of the point from its nearest center.'''
            centers = []
            X = np.array(self.features)

            # Sample the first point
            initial_index = np.random.choice(range(X.shape[0]), )
            centers.append(X[initial_index, :].tolist())

            print('max: ', np.max(np.sum((X - np.array(centers)) ** 2)))

            # Loop and select the remaining points
            for i in range(self.n_clusters - 1):
                print(i)
                distance = self.dist(X, np.array(centers))

                if i == 0:
                    pdf = distance / np.sum(distance)
                    centroid_new = X[np.random.choice(range(X.shape[0]), replace=False, p=pdf.flatten())]
                else:
                    # Calculate the distance of each point from its nearest centroid
                    dist_min = np.min(distance, axis=1)
                    pdf_method = True
                    if pdf_method:
                        pdf = dist_min / np.sum(dist_min)
                        # Sample one point from the given distribution
                        centroid_new = X[np.random.choice(range(X.shape[0]), replace=False, p=pdf)]
                    else:
                        index_max = np.argmax(dist_min, axis=0)
                        centroid_new = X[index_max, :]
                centers.append(centroid_new.tolist())
            for i, c in enumerate(centers):
                centroids[i] = c
        else:
            raise ValueError("init_type must be: random or k-means++")

        return centroids

    @staticmethod
    def dist(data, centers):
        distance = np.sum((np.array(centers) - data[:, None, :]) ** 2, axis=2)
        return distance

    def fit(self, show_logs=True):
        """
        Authors method of Kmeans algorithm. Steps:
        1. Select random centroid for each cluster
        2. Assign all the points to the closest cluster centroid
        3. Recompute centroids of newly formed clusters
        4. Check if intracluster distances exceeds tolerance threshold
        5. Repeat steps 2 and 3 until stop criteria is met
        6. Repeat all above points (one Kmeans algorithm run) n times (with different centroid seeds) to get the best result in terms of inertia
        :return:
        """
        tolerance = 0.0001
        best_inertia = -1
        best_classifications = None
        best_temp = None
        # 6. Repeat all above points (one Kmeans algorithm run) n times (with different centroid seeds) to get the best result in terms of inertia
        for r in range(self.n_init):
            # 1. Select centroids for each cluster
            centroids = self.initialize_centroids(init_type=self.init_type)

            # 5. Repeat steps 2 and 3 until stop criteria is met
            for i in range(self.max_iter):
                if show_logs:
                    print("iteration {}".format(i))
                # 2. Assign all the points to the closest cluster centroid
                classifications = {}
                temp = {}
                for j in range(self.n_clusters):
                    classifications[j] = []
                    temp[j] = []
                for j, fs in enumerate(self.features):
                    # print("j ", j)
                    distances = [self.euclidean_distance(fs, centroids[centroid]) for centroid in centroids]
                    # input(["distances ", distances])
                    classification = distances.index(min(distances))
                    classifications[classification].append(fs)
                    temp[classification].append(j)

                # 3. Recompute centroids of newly formed clusters
                prev_centroids = dict(centroids)
                for classification in classifications:
                    centroids[classification] = self.average(classifications[classification])

                if show_logs:
                    print("classifications ")
                    for i in temp:
                        print(i, temp[i])
                # 4. Check if intracluster distances exceeds tolerance threshold
                optimized = True
                for c in centroids:
                    original_centroid = prev_centroids[c]
                    current_centroid = centroids[c]
                    difference = 0
                    for j in range(len(current_centroid)):
                        difference += (current_centroid[j] - original_centroid[j])
                    if difference > tolerance:
                        if show_logs:
                            print("difference = {}".format(difference))
                        optimized = False

                if optimized:
                    break

            inertia = self.inertia(classifications=classifications, centroids=centroids)
            if best_inertia == -1 or inertia < best_inertia:
                best_inertia = inertia
                best_classifications = classifications
                best_temp = temp

        print("Best classification")
        for i in best_temp:
            print(i, best_temp[i])
        print("best_inertia = {}".format(best_inertia))
        # Zwroc labels, czyli np. [0, 0, 1, 1] oznacza, ze pierwsze 2 docsy sa w klastrze 0, kolejne 2 w klastrze 1
        labels = []
        for fs in self.features:
            for c in best_classifications:
                if fs in best_classifications[c]:
                    labels.append(c)

        return labels

    def validate_clustering(self):
        """
        Print validation scores for clustering
        :return validation_scores (dict):       dict with values of validatation scores
        """
        reference_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        validation_scores = {"silhouette_score": silhouette_score(X=self.features, labels=self.labels),
                             "v_measure_score": v_measure_score(
                                 labels_true=reference_labels, labels_pred=self.labels),
                             "adjusted_rand_score": adjusted_rand_score(
                                 labels_true=reference_labels, labels_pred=self.labels),
                             "davies_bouldin_score": davies_bouldin_score(X=self.features, labels=self.labels)
                             }
        print("### Clustering validation scores ###")
        if self.labels is not []:
            print("silhouette_score: ", validation_scores["silhouette_score"])
            print("v_measure_score: ", validation_scores["v_measure_score"])
            print("adjusted_rand_score: ", validation_scores["adjusted_rand_score"])
            print("davies_bouldin_score: ", validation_scores["davies_bouldin_score"])
        else:
            raise TypeError("Labels are not set")
        return validation_scores

    @staticmethod
    def euclidean_distance(p1, p2):
        """
        Calculate euclidean distance between points: p1 and p2.
        :param p1 (list[float]):    point 1
        :param p2 (list[float]):    point 2
        :return d (float):          euclidean distance
        """
        if not (len(p1) == len(p2)):
            raise TypeError("Point must be lists of floats with the same lengths")
        d = 0
        for i in range(len(p1)):
            d += (p1[i] - p2[i]) ** 2

        return sqrt(d)

    @staticmethod
    def average(vectors):
        """
        Calculate average values for list of vectors.
        :return avg(list[float]):   average vector
        """
        # print("vectors")
        # print(vectors)
        # input("here")

        avg = [0 for i in range(len(vectors[0]))]
        for i in range(len(vectors)):
            for j in range(len(vectors[i])):
                avg[j] += vectors[i][j]

        avg = [avg[i] / len(vectors) for i in range(len(vectors[0]))]
        # print("avg ", avg)

        return avg

    def inertia(self, classifications, centroids):
        """
        Calculate inertia which represents the sum of squared intracluster distances.
        :param classifications (dict):   classification of points to clusters, e.g. {0: [fs1, fs2], 1: [fs3], 2: [fs4, fs5]}
        :param centroids (dict):         centroids of clusters, e.g. {0: c1, 1: c2}, where c1 is vector represents point (centroid) position
        :return inertia (float):         sum of squared intracluster distances
        """
        inertia = 0
        for centroid in centroids:
            intra_distance = 0
            for fs in classifications[centroid]:
                intra_distance += self.euclidean_distance(centroids[centroid], fs) ** 2
            inertia += intra_distance

        print("inertia = {}".format(inertia))
        return inertia

    def determine_optimal_number_of_clusters(self, method='elbow_method'):
        """
        Call proper method to determine proper number of clusters.
        :param method (str):        method to call, e.g. elbow_method, silhoulette_method, davies_bouldin_method
        :return:
        """
        if method not in ['elbow_method', 'silhoulette_method', 'davies_bouldin_method']:
            raise ValueError("Method must be: elbow_method, silhoulette_method or davies_bouldin_method")

        method_to_call = getattr(Clustering, method)
        method_to_call(self)

    def elbow_method(self):
        """
        Elbow method for determining optimal number of clusters
        :return:
        """
        distortions = []
        if len(self.features) - 1 >= 10:
            k_max = 10
        else:
            k_max = len(self.features) - 1
        K = range(1, k_max)
        for k in K:
            kmeanModel = KMeans(n_clusters=k, n_init=self.n_init).fit(self.features)
            distortions.append(kmeanModel.inertia_)

        plt.figure(figsize=(16, 8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    def silhoulette_method(self):
        """
        Silhoulette method for determining optimal number of clusters
        :return:
        """
        sil = []
        if len(self.features) - 1 >= 10:
            k_max = 10
        else:
            k_max = len(self.features) - 1
        K = range(2, k_max)

        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in K:
            kmeans = KMeans(n_clusters=k, n_init=self.n_init).fit(self.features)
            labels = kmeans.labels_
            sil.append(silhouette_score(self.features, labels, metric='euclidean'))

        plt.figure(figsize=(16, 8))
        plt.plot(K, sil, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhoulette score')
        plt.title('Silhoulette score showing the optimal k')
        plt.show()

    def davies_bouldin_method(self):
        """
        Davies-Bouldin method for determining optimal number of clusters
        :return:
        """
        scores = []
        if len(self.features) - 1 >= 10:
            k_max = 10
        else:
            k_max = len(self.features) - 1
        K = range(2, k_max)
        for k in K:
            kmeans = KMeans(n_clusters=k, n_init=self.n_init)
            model = kmeans.fit_predict(self.features)
            score = davies_bouldin_score(self.features, model)
            scores.append(score)

        plt.plot(K, scores, linestyle='--', marker='o', color='b')
        plt.xlabel('K')
        plt.ylabel('Davies Bouldin score')
        plt.title('Davies Bouldin score vs. K')
        plt.show()
