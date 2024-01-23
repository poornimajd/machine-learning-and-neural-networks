import numpy as np

from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist

#################################################################
# Load Dataset
#################################################################

dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    shuffle=True,
    random_state=42,
)

labels = dataset.target
unique_labels, category_sizes = np.unique(labels, return_counts=True)
true_k = unique_labels.shape[0]

#################################################################
# Evaluate Fitness
#################################################################
def fit_and_evaluate(km, X, n_runs=5):

    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        km.fit(X)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")

#################################################################
# Vectorize
#################################################################
vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)

X_tfidf = vectorizer.fit_transform(dataset.data)

#################################################################
# (TODO): Implement K-Means
#################################################################

class KMeans:
    # labels_ = [] #Todo
    def __init__(self, n_clusters=20, max_iter=50, tol=0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None
    # def compute_distances(self,X_dense, centroids):
    #   distances = np.zeros((X_dense.shape[0], centroids.shape[0]))
    #   for i, centroid in enumerate(centroids):
    #       diff = X_dense - centroid  # Difference between each point and the centroid
    #       sq_diff = np.square(diff)  # Squared differences
    #       sum_sq_diff = np.sum(sq_diff, axis=1)  # Sum of squared differences for each point
    #       distances[:, i] = np.sqrt(sum_sq_diff)  # Square root to get Euclidean distance
    #   return distances

    def fit(self, X):
        # Randomly initialize centroids
        # idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        # self.centroids = X[idx].toarray()  # Ensure centroids are 2D
        # np.random.seed(2) 
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        selected_indices = indices[:self.n_clusters]
        self.centroids = X[selected_indices].toarray()

        for _ in range(self.max_iter):
            # Convert X to dense array if it's sparse
            X_dense = X.toarray() if hasattr(X, "toarray") else X

            # Assign labels based on closest centroid
            distances = cdist(X_dense, self.centroids, 'euclidean')
            # distances = self.compute_distances(X_dense, self.centroids)
            self.labels_ = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_points = X_dense[self.labels_ == i]
                if len(cluster_points) > 0:
                    new_centroids.append(np.mean(cluster_points, axis=0))
                else:
                    # Handle empty clusters
                    new_centroids.append(self.centroids[i])

            new_centroids = np.array(new_centroids)

            # Check for convergence
            if np.sum((new_centroids - self.centroids) ** 2) < self.tol:
                # print("done")
                break

            self.centroids = new_centroids
        # print("iter",_)

    def set_params(self, random_state=None):
        np.random.seed(random_state)

# Use KMeans
kmeans = KMeans()
fit_and_evaluate(kmeans, X_tfidf)
