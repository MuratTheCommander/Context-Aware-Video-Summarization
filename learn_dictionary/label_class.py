from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np


n_clusters = 120

def set_vocabulary(stip_features):

    all_descriptors = []


    for feature in stip_features:
        all_descriptors.append(feature.descriptor)
    
    all_descriptors = np.array(all_descriptors) 

    kmeans = KMeans(n_clusters=n_clusters,random_state=42)

    kmeans.fit(all_descriptors)

    vocabulary = kmeans.cluster_centers_

    return vocabulary


def assign_visual_words(stip_features,vocabulary):

    if not stip_features:
        return stip_features #Empty list, nothing to assign
    
    #1. stack all descriptors into a matrix
    descriptors = np.array([f.descriptor for f in stip_features]) # shape: (n_samples,162)

    #2. compute distances to all cluster centers
    distances = cdist(descriptors,vocabulary,metric='euclidean') #shape: (n_samples,n_clusters)

    #3. Find the nearest cluster center for each descriptor
    nearest_indices = np.argmin(distances,axis=1)

    #4. Assign the nearest cluster label to each STIPFeature object
    for feature, cluster_idx in zip(stip_features,nearest_indices):
        feature.class_label = cluster_idx

    return stip_features




