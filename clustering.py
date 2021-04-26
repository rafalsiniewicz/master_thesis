from sklearn.metrics import silhouette_score, adjusted_rand_score
from base_class import *
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, features=[], max_iter=100, n_clusters=3):
        self.features: List[float] = features               # features for clustering
        self.max_iter: int = max_iter
        self.clustering_method = None
        self.n_clusters = n_clusters
        self.labels = None

    def run(self):
        """
        Cluster features data.
        :return:
        """
        self.clustering_method = KMeans(n_clusters=self.n_clusters, random_state=0, max_iter=self.max_iter).fit(self.features)
        self.labels = self.clustering_method.labels_
        print("kmeans.labels_", self.clustering_method.labels_)
        print("kmeans.n_iter_", self.clustering_method.n_iter_)


    def validate_clustering(self):
        silhouette = silhouette_score(self.features, self.clustering_method.labels_).round(2)
        print("silhouette: ", silhouette)
        # ari = adjusted_rand_score([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], self.clustering_method.labels_)
        # print("ari", ari)