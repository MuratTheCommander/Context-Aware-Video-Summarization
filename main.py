import numpy as np
from preprocessing.main import process_video
from learn_dictionary.label_class import set_vocabulary, assign_visual_words
from learn_dictionary.initial_graph import build_edges, build_Graph
from learn_dictionary.correlation_matrices import (
    build_adjacency_matrix_M,
    build_segment_laplacian
)
from learn_dictionary.learn_dictionaries import initialize_Df
from main_utils import *  # newly added wrapper
from preprocessing.stip import detect_STIPs
from preprocessing.hog_hof import compute_HOG_HOF
from online_update.online_optimizers import (
    optimize_Df_online,
    sparse_code_online
)
from online_update.graph_process import *
from learn_dictionary import *

# Step 1: Load video and extract segment-level features
video_path = r'C:\Users\lenovo\Desktop\KFAU\Graduation Project\code\CAVS\VIRAT_S_000200_03_000657_000899.mp4'

feature_matrix_X, training_video_segments, testing_video_segments,test_mask_segments, stip_to_segment, stip_features, _ = process_video(video_path=video_path)

# Step 2: Create visual vocabulary and assign visual words
vocabulary = set_vocabulary(stip_features)
stip_features = assign_visual_words(stip_features, vocabulary)

# Step 3: Build STIP-level graph
class_cooccurrence = {}
class_total_neighbors = {}
spatial_threshold = 100
temporal_threshold = 60

edges, class_cooccurrence, class_total_neighbors = build_edges(
    stip_features,
    spatial_threshold,
    temporal_threshold,
    class_cooccurrence,
    class_total_neighbors
)

G0 = build_Graph(stip_features, edges, name="G0")

Dg = []
Dg.append(G0)

# Step 4: Build Laplacian matrix for segments
adjacency_matrix_M, _ = build_adjacency_matrix_M(G0, class_cooccurrence, class_total_neighbors)
laplacian_matrix_L, _ = build_segment_laplacian(adjacency_matrix_M, stip_to_segment, feature_matrix_X.shape[1])

# Step 5: Initialize dictionary
dF = initialize_Df(feature_matrix_X)

# Step 6: Train Df and B using alternating minimization (Eq. 2)
alpha1 = 0.01  # Laplacian trace regularization
alpha2 = 0.1   # group sparsity
alpha3 = 0.1   # element-wise sparsity

Df_final, B_final = train_Df_B(
    X=feature_matrix_X,
    Df_init=dF,
    Lp=laplacian_matrix_L,
    alpha1=alpha1,
    alpha2=alpha2,
    alpha3=alpha3,
    max_outer_iter=10,
    max_inner_iter=15,
    tol=1e-4
)

print("Training completed.")
print(f"Final Df shape: {Df_final.shape}, Final B shape: {B_final.shape}")

#Df_copy = Df_final.copy()

summary = []

K = Df_final.shape[1] # Number of atoms

d = Df_final.shape[0] # Feature Dimension

segment_length = 90

#Online Update  starts from now on

P = np.zeros((K,K)) # ∑ B Bᵀ

Q = np.zeros((d,K)) # ∑ X Bᵀ

test_segmnet_id = 0

reconstruction_threshold = 100 

first_graph_comparsion_threshold = 100

for seg_idx , (test_segment,test_mask_segment) in enumerate (zip(testing_video_segments,test_mask_segments)):

    is_segment_added_to_summary = False

    test_segment_id += 1

    g_test_class_cooccurrence = {}
    g_test_class_total_neighbors = {}
    

    stip_points,stip_features = detect_STIPs(test_segment,test_mask_segment,seg_idx*segment_length)

    hist = compute_HOG_HOF(test_segment,stip_points) 

    x_test = hist.reshape(-1,1) #shape (d,1) Xtest that represents the feature vector for the current test segment

    b_test = sparse_code_online(x_test,Df_final)

    reconstruction_error = np.linalg.norm(x_test - Df_final@b_test)**2

    if reconstruction_error > reconstruction_threshold:

        summary.append(test_segment)
        is_segment_added_to_summary = True

        P += b_test @ b_test.T

        Q += x_test @ b_test.T  
    
        Df_final = optimize_Df_online(P,Q,Df_final)

        print(f"Df is updated for the segment {seg_idx}")
    
    g_test_edges, g_test_class_cooccurrence, g_test_class_total_neighbors = build_edges(
        stip_features,spatial_threshold,temporal_threshold,g_test_class_cooccurrence,g_test_class_total_neighbors
    )

    g_test = build_Graph(stip_features,g_test_edges,f"g{seg_idx}")

    candidates = []
    best_h_score = float('inf')
    best_match_graph = None

    h_threshold = 50 # Change this number based on how strict you want to be

    for g_to_compare in Dg:
      
      # First filter: fast similarity check
      comparison_score = compute_graph_similarity(g_test,g_to_compare)

      if comparison_score > first_graph_comparsion_threshold:

            candidates.append(g_to_compare)

            # Now do semantic matching with Hungarian
            H = construct_H_matrix(g_test,g_to_compare)
            h_score = compute_h_score(H)

            print(f"Semantic match score (h) with {g_to_compare.name}: {h_score:.2f}")

            if h_score < best_h_score:
             best_h_score = h_score
             best_match_graph = g_to_compare

    # Decision: Add g_test or not
    if not candidates:
        Dg.append(g_test)
        print(f"No similar graph. Added {g_test.name} to Dg.")
        if is_segment_added_to_summary:
            continue
        else:
            summary.append(test_segment)
    elif best_h_score > h_threshold:
         Dg.append(g_test)
         print(f"No good semantic match. Added {g_test.name} to Dg.")
         if is_segment_added_to_summary:
            continue
         else:
            summary.append(test_segment)
    else:
         print(f"{g_test.name} is semantically similar to {best_match_graph.name} (h = {best_h_score:.2f})")
         merge_graphs(g_test,best_match_graph)










     






























