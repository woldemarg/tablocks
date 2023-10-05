from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score
from supervenn import supervenn
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.ensemble import RandomForestClassifier
from prosphera.projector import Projector
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# %%

n_components = 3
n_clusters = 3


# %%

# X, y = load_wine(return_X_y=True, as_frame=True)

# %%

X, y = make_classification(
    n_samples=5000,
    n_features=15,
    n_informative=n_components,
    n_classes=n_clusters,
    n_clusters_per_class=1,
    class_sep=2,
    random_state=1234)

# %%

scaler = StandardScaler().set_output(transform="pandas")
scaled_X = scaler.fit_transform(X)

# %%

scaled_pca = PCA(n_components=n_components).fit(scaled_X)
expld_vars = scaled_pca.explained_variance_ratio_.reshape(-1, 1)
components = np.abs(scaled_pca.components_)

np.cumsum(expld_vars.ravel())

# %%

# d = pd.DataFrame(components).apply(lambda x:
#                                    normalize(x.to_numpy().reshape(1, -1), norm='l1')[0], axis=1)

# %%

# d = normalize(pd.DataFrame(components), norm='l1', axis=1)

d = pd.DataFrame(components).rank(pct=True, axis=1)

fimp = normalize(d.sum(0).to_numpy().reshape(1, -1), norm='l1')
# fimp =d.sum(0)

# %%

# fimp = normalize(
# np.sum(components * expld_vars, axis=0).reshape(1, -1),
# norm='l1')[0]

# fimp = np.sum(components * expld_vars, axis=0).reshape(1, -1)
# fimp =  np.clip(fimp, a_min=0.01, a_max=None)


# %%

X_norm = normalize(scaled_X)
# X_norm_w = normalize(scaled_X * fimp.reshape(1, -1))
X_norm_w = X_norm * fimp.reshape(1, -1)


# selector = VarianceThreshold()
# selector.fit(X_norm)
# fimp = selector.variances_
# fimp = normalize(selector.variances_.reshape(1, -1), norm='l1')
# fimp.sum()
# %%

# centroids_x = pd.DataFrame(X_norm).groupby(y).mean()

# %%

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=1234,
    n_init='auto').fit(X_norm)


labels = kmeans.labels_
centroids = kmeans.cluster_centers_
centroids_y = pd.DataFrame(X_norm).groupby(y).mean()


centroids_adj = pairwise_distances_argmin(
    centroids,
    centroids_y,
    axis=1,
    metric='cosine')

labels = (pd.Series(labels)
          .replace(dict(
              zip(
                  range(n_clusters),
                  centroids_adj))))


# %%

kmeans_w = KMeans(
    n_clusters=n_clusters,
    random_state=1234,
    n_init='auto').fit(X_norm_w)

labels_w = kmeans_w.labels_
centroids_w = kmeans_w.cluster_centers_
centroids_wy = pd.DataFrame(X_norm_w).groupby(y).mean()

centroids_w_adj = pairwise_distances_argmin(
    centroids_w,
    centroids_wy,
    axis=1,
    metric='cosine')

labels_w = (pd.Series(labels_w)
            .replace(dict(
                zip(
                    range(n_clusters),
                    centroids_w_adj))))

# %%

mtx = 1 - pairwise_distances(
    X_norm_w,
    metric='cosine')

mtx = np.clip(mtx, a_min=0, a_max=None)

spectra = SpectralClustering(n_clusters=n_clusters,
                             affinity='precomputed',
                             random_state=1234).fit(mtx)

labels_s = spectra.labels_
centroids_s = pd.DataFrame(X_norm_w).groupby(labels_s).mean()

centroids_s_adj = pairwise_distances_argmin(
    centroids,
    centroids_s,
    axis=1,
    metric='cosine')

labels_s = (pd.Series(labels_s)
            .replace(dict(
                zip(
                    range(n_clusters),
                    centroids_s_adj))))


# %%

visualizer.project(
    data=X_norm_w,
    labels=labels_w
)

# %%

visualizer.project(
    data=X_norm,
    labels=labels
)

# %%

visualizer.project(
    data=X_norm_w,
    labels=labels_s
)

# %%

visualizer.project(
    data=X,
    labels=y
)


# %%

clf = RandomForestClassifier(random_state=1234, n_jobs=-1)

clf.fit(X, y)


pd.Series(clf.feature_importances_).sort_values().plot.barh()


pd.Series(np.abs(fimp.ravel())).sort_values().plot.barh()

# %%

visualizer = Projector()

# %%

visualizer.project(
    data=X_norm_w,
    labels=labels_w
)


# %%

visualizer.project(
    data=X_norm,
    labels=labels)

# %%


# %%
#
# np.where(y ==0)[0]

sets = [set(np.where(s == 2)[0]) for s in (y, labels, labels_w)]
lab = ['tru', 'base', 'norm']
supervenn(sets, lab)

# %%


# %%
balanced_accuracy_score(y, labels)
balanced_accuracy_score(y, labels_w)
