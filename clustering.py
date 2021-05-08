from sklearn.metrics import silhouette_score, adjusted_rand_score
from base_class import *
from sklearn.cluster import KMeans
from math import sqrt
from random import randint


class Clustering:
    def __init__(self, features=[], max_iter=100, n_clusters=3, n_init=10, init_type='random'):
        self.features: List[List[float]] = features  # list of features for clustering
        self.max_iter: int = max_iter # Maximum number of iterations of the k-means algorithm for a single run.
        self.clustering_method = None
        self.n_clusters: int = n_clusters # The number of clusters to form as well as the number of centroids to generate
        self.n_init: int = n_init  # Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
        self.labels = None
        self.init_type: str = init_type

    def run(self, sklearn_method=True):
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
            self.clustering_method = self.fit()
            self.labels = self.clustering_method
            print("labels: ", self.labels)

    def fit(self):
        """
        Authors method of Kmeans algorithm. Steps:
        1. Select random centroid for each cluster
        2. Assign all the points to the closest cluster centroid
        3. Recompute centroids of newly formed clusters
        4. Repeat steps 2 and 3 until stop criteria is met
        5. Repeat all above points (one Kmeans algorithm run) n times (with different centroid seeds) to get the best result in terms of inertia
        :return:
        """
        tolerance = 0.0001
        best_inertia = -1
        best_classifications = None
        best_temp = None
        for r in range(self.n_init):
            # 1. Select random centroid for each cluster
            centroids = {}
            selected_centroids = []
            for i in range(self.n_clusters):
                # centroids[i] = self.features[i]
                rand_fs_idx = randint(0, len(self.features) - 1)  # random feature set index
                while rand_fs_idx in selected_centroids:
                    rand_fs_idx = randint(0, len(self.features) - 1)
                centroids[i] = self.features[rand_fs_idx]
                selected_centroids.append(rand_fs_idx)

            for i in range(self.max_iter):
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

                print("classifications ")
                for i in temp:
                    print(i, temp[i])
                optimized = True
                for c in centroids:
                    original_centroid = prev_centroids[c]
                    current_centroid = centroids[c]
                    difference = 0
                    for j in range(len(current_centroid)):
                        difference += (current_centroid[j] - original_centroid[j])
                    if difference > tolerance:
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
        silhouette = silhouette_score(self.features, self.clustering_method.labels_).round(2)
        print("silhouette: ", silhouette)
        # ari = adjusted_rand_score([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], self.clustering_method.labels_)
        # print("ari", ari)

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
        print("avg ", avg)

        return avg

    def inertia(self, classifications, centroids):
        """
        Calculate inertia which represents the sum of intracluster distances.
        :param classifications (dict):   classification of points to clusters, e.g. {0: [fs1, fs2], 1: [fs3], 2: [fs4, fs5]}
        :param centroids (dict):         centroids of clusters, e.g. {0: c1, 1: c2}, where c1 is vector represents point (centroid) position
        :return inertia (float):         sum of intracluster distances
        """
        inertia = 0
        for centroid in centroids:
            intra_distance = 0
            for fs in classifications[centroid]:
                intra_distance += self.euclidean_distance(centroids[centroid], fs)**2
            inertia += intra_distance

        print("inertia = {}".format(inertia))
        return inertia

