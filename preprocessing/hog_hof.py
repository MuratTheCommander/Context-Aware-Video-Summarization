from skimage.feature import hog
import cv2 as cv
import numpy as np


def compute_HOG_HOF(segment, stip_points, spatial_bins=(2,2), temporal_bins=2):
    """
    Compute HOG (appearance) and HOF (motion) features for a segment,
    using spatio-temporal pyramids.
    Returns concatenated HOG + HOF features across all bins.
    """
    # Get frame size and time length
    frame_height, frame_width = segment[0].shape[:2]
    segment_length = len(segment)

    # Bin sizes
    bin_w = frame_width // spatial_bins[0]
    bin_h = frame_height // spatial_bins[1]
    bin_t = segment_length // temporal_bins

    # Prepare bins
    bins = {}
    for x in range(spatial_bins[0]):
        for y in range(spatial_bins[1]):
            for t in range(temporal_bins):
                bins[(x, y, t)] = []  # each bin stores stip_points

    # Assign STIPs to bins
    for stip in stip_points:
        x_coord, y_coord, t_coord = stip
        x_bin = min(int(x_coord) // bin_w, spatial_bins[0]-1)
        y_bin = min(int(y_coord) // bin_h, spatial_bins[1]-1)
        t_bin = min(int(t_coord) // bin_t, temporal_bins-1)
        bins[(x_bin, y_bin, t_bin)].append((x_coord, y_coord, t_coord))

    feature_vector = []

    for bin_key, bin_stips in bins.items():
        hog_features = []
        hof_features = []

        for stip in bin_stips:
            x, y, t = map(int, stip)

            if t >= len(segment) - 1:
                continue  # skip last frame for HOF

            # Extract patch around STIP
            frame = segment[t]
            patch = frame[y-16:y+16, x-16:x+16]
            if patch.shape == (32, 32, 3):
                hog_feature = hog(
                    patch,
                    orientations=8,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1),
                    channel_axis=-1
                )
                hog_features.append(hog_feature)

            # Optical flow for HOF
            prev_frame = cv.cvtColor(segment[t], cv.COLOR_BGR2GRAY)
            next_frame = cv.cvtColor(segment[t+1], cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(
                prev_frame, next_frame, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                dx, dy = flow[y, x]
                angle = np.arctan2(dy, dx)
                hof_features.append(angle)

        # Pool features inside this bin
        hog_hist = np.mean(hog_features, axis=0) if hog_features else np.zeros(128)
        hof_hist, _ = np.histogram(hof_features, bins=8, range=(-np.pi, np.pi))

        # Concatenate HOG and HOF for this bin
        bin_feature = np.concatenate([hog_hist, hof_hist])
        feature_vector.append(bin_feature)

    # Flatten all bins together
    feature_vector = np.concatenate(feature_vector)
    return feature_vector