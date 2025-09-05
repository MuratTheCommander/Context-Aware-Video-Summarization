def assign_visual_words(stip_features, vocabulary, threshold_factor=0.4):

    if not stip_features:
        return stip_features, vocabulary  # Return both

    # Step 1: Stack all descriptors into a matrix
    descriptors = np.array([f.descriptor for f in stip_features])  # shape: (n_samples,162)

    # Step 2: Compute distances to all cluster centers
    distances = cdist(descriptors, vocabulary, metric='euclidean')  # shape: (n_samples, n_clusters)

    # Step 3: Find the nearest cluster center and distance
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = np.min(distances, axis=1)

    # Step 3.5: Compute distance threshold based on inter-center distances
    inter_center_distances = cdist(vocabulary, vocabulary, metric='euclidean')
    max_inter_center_distance = np.max(inter_center_distances)
    threshold = threshold_factor * max_inter_center_distance

    # Step 4: Assign class labels (with dynamic update for outliers)
    for i, feature in enumerate(stip_features):
        if nearest_distances[i] > threshold:
            # Add new cluster center
            vocabulary = np.vstack([vocabulary, feature.descriptor])
            feature.class_label = vocabulary.shape[0] - 1  # new label
        else:
            feature.class_label = nearest_indices[i]

    return stip_features, vocabulary
