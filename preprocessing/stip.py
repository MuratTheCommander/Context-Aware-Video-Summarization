import cv2 as cv
from skimage.feature import hog
import numpy as np


class STIPFeature:

    id = 0

    def __init__(self,x,y,t,descriptor,class_label=None):    
        self.x = x
        self.y = y
        self.t = t
        self.descriptor = descriptor
        self.class_label = class_label
        self.id = STIPFeature.id

        STIPFeature.id += 1

def detect_STIPs(segment, mask_segment,start_frame=0):
    """
    Simplified STIP detection using Harris corners + optical flow tracking.
    Returns STIP coordinates (x, y, t) for the segment.
    """
    stip_points = []
    stip_features = []

    frame_height, frame_width = segment[0].shape[:2]

    for t in range(len(segment) - 1):
        frame = segment[t]
        mask = mask_segment[t]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        masked_gray = cv.bitwise_and(gray, gray, mask=mask)

        corners = cv.goodFeaturesToTrack(
            masked_gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=7
        )

        if corners is not None:
            next_gray = cv.cvtColor(segment[t+1], cv.COLOR_BGR2GRAY)
            next_masked_gray = cv.bitwise_and(next_gray, next_gray, mask=mask_segment[t+1])

            next_corners, status, _ = cv.calcOpticalFlowPyrLK(
                masked_gray, next_masked_gray,
                corners, None,
                winSize=(15, 15),
                maxLevel=2,
            )

            if next_corners is None or status is None:
                continue

            for i in range(len(corners)):
                if status[i].item():  # or status[i][0]
                    x, y = corners[i][0]
                    global_t = t + start_frame

                    stip_points.append((x,y,global_t))

                    if not (16 <= int(x) < frame_width-16 and 16 <= int(y) < frame_height-16):
                        continue

                    patch = frame[int(y)-16:int(y)+16, int(x)-16:int(x)+16]
                    if patch.shape != (32, 32, 3):
                        continue

                    hog_feature = hog(
                        patch,
                        orientations=8,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1),
                        channel_axis=-1
                    )

                    # Optical flow for HOF
                    prev_frame = cv.cvtColor(segment[t], cv.COLOR_BGR2GRAY)
                    next_frame = cv.cvtColor(segment[t+1], cv.COLOR_BGR2GRAY)
                    flow = cv.calcOpticalFlowFarneback(
                        prev_frame, next_frame, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )

                    if 0 <= int(y) < flow.shape[0] and 0 <= int(x) < flow.shape[1]:
                        dx, dy = flow[int(y), int(x)]
                        angle = np.arctan2(dy, dx)
                        hof_hist, _ = np.histogram([angle], bins=8, range=(-np.pi, np.pi))
                    else:
                        hof_hist = np.zeros(8)

                    descriptor = np.concatenate([hog_feature, hof_hist])

                    stip_feature = STIPFeature(x, y, global_t, descriptor)
                    stip_features.append(stip_feature)

    return stip_points,stip_features

