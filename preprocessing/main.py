from .stip import detect_STIPs
from .hog_hof import compute_HOG_HOF
import numpy as np
import cv2 as cv

def process_video(video_path):

  cap = cv.VideoCapture(video_path)
  fgbg = cv.createBackgroundSubtractorMOG2()
  frame_count = 0
  frames = []
  masks = []
  while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    frame_count += 1
    fgmask = fgbg.apply(frame)
    masks.append(fgmask)   #saving masks for segmentation and feature extraction

  cap.release() 

  print(f"Total of {len(frames)} frames retrieved")
  print(f"Total of {len(masks)} masks retrieved")

  assert len(frames) == len(masks)

  segment_length = 90

  video_segments = [frames[i:i+segment_length] for i in range(0,len(frames),segment_length)]
  mask_segments = [masks[i:i+segment_length] for i in range(0,len(masks),segment_length)]

  print(f"Total segments: {len(video_segments)}")

  if video_segments and mask_segments:
    print(f"First segment: {len(video_segments[0])} frames, {len(mask_segments[0])} masks")
  else:
    print("No segments found.Check if the video is valid and non-empty.")

  train_ratio = 0.33  
  split_index = int(len(video_segments)*train_ratio)

  train_video_segments = video_segments[:split_index]
  train_mask_segments = mask_segments[:split_index]

  test_video_segments = video_segments[split_index:]
  test_mask_segments = mask_segments[split_index:]

  print(f"Training segments: {len(train_video_segments)}")
  print(f"Testing segments: {len(test_video_segments)}")

  
  feature_histograms = []
  stip_all_training = []
  stip_to_segment = {}

  stip_all_testing = []

  for seg_idx, (vid_seg,mask_seg)  in enumerate(zip(train_video_segments,train_mask_segments)):
    # Detect STIPs in the segment
    stip_points,stip_features = detect_STIPs(vid_seg,mask_seg,start_frame=seg_idx*segment_length)
    stip_all_training.append(stip_points)

    for stip_feature in stip_features:
      stip_to_segment[stip_feature.id] = seg_idx #store mapping from STIP id to segment id
      

    # Compute HOG + HOF features
    hist = compute_HOG_HOF(vid_seg,stip_points)
    feature_histograms.append(hist)

    print(f"Training Segment {seg_idx}: {len(stip_points)} STIPs | Features shape: {hist.shape}")

  training_feature_matrix_X = np.vstack(feature_histograms).transpose()

  """ for seg_idx, (test_seg,test_mask_seg) in enumerate(zip(test_video_segments,test_mask_segments)):
    stip_points,stip_features = detect_STIPs(test_seg,test_mask_seg,start_frame=segment_length)
    stip_all_testing.append(stip_points)

    hist = compute_HOG_HOF(test_seg,stip_points)
    feature_histograms.append(hist)

    print(f"Training Segment {seg_idx}: {len(stip_points)} STIPs | Features shape: {hist.shape}")

    testing_feature_matrix_X = np.vstack(feature_histograms).transpose()"""



  return training_feature_matrix_X,train_video_segments,test_video_segments,test_mask_segments,stip_to_segment,stip_features,split_index