import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.cluster import DBSCAN

class Coin:
  x = 0
  y = 0

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def explain(self, out_num):
    x = self.x 
    y = self.y
    inliners = x[y == 1]
    outliers = x[y == -1]

    # Resample outlier to form cluster.
    n_dimens = len(outliers[0])
    n_samples = 5
    resampled = np.random.normal(outliers[out_num].tolist(), [0.01 for _ in range(n_dimens)], (n_samples, n_dimens))
    np.append(resampled, outliers[out_num,:])

    # Find context of outlier.
    n_neigh = 30
    nbrs = NearestNeighbors(n_neighbors=n_neigh, algorithm='kd_tree', metric='euclidean').fit(inliners)
    distances, neighbors = nbrs.kneighbors(outliers)

    # Clustering outlier context.
    out_neigh = neighbors[out_num]
    context = inliners[out_neigh,:]
    db = DBSCAN(eps=0.1, min_samples=3).fit(context)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters_ == 0:
      return "Could not find enough clusters in outlier context."

    # Mask for outlier neighbors' clusters.
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    n_features = len(x[0])
    s_il = np.empty([n_clusters_, n_features])
    cluster_sizes = np.empty([n_clusters_])
    unique_labels = set(labels)

    for l in unique_labels:
      if l != -1:
        class_member_mask = (labels == l)
        cluster = context[class_member_mask & core_samples_mask]
        cluster_sizes[l] = (len(cluster))

        # Compute coefficients.
        X = np.concatenate((cluster, resampled), axis=0)
        y = [1 if i < len(cluster) else 0 for i in range(len(X))]
        clf = LinearSVC(penalty="l1", dual=False, random_state=0, tol=1e-5)
        clf.fit(X, y) 
        
        # Find nearest neighbors in cluster.
        # should nearest neighbor be form the cluster?
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', metric='euclidean').fit(cluster)
        cluster_nbrs = nbrs.kneighbors(cluster, return_distance=False)[:,1] # [0] -- the same point

        for m in range(n_features):
          dist = [abs(context[n][m] - cluster[i][m]) for i, n in enumerate(cluster_nbrs)]
          gamma = sum(dist) / len(cluster)
          s_il[l][m] = abs(clf.coef_[0][m]) / gamma

      importance = np.empty([n_features])
      
      for m in range(n_features):
        s_sum = 0
        for l in unique_labels:
          if l != -1:
            s_sum += cluster_sizes[l] * s_il[l][m]
        importance[m] = s_sum / n_neigh
    return importance
