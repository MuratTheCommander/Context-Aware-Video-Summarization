import numpy as np
from initial_graph import gaussian

video_frame_width = 640  ###  use these values as default values
mu_s = 0.1*video_frame_width
sigma_s = 0.1*video_frame_width
mu_t = 90
sigma_t = 30



def build_adjacency_matrix_M(G0,class_cooccurrence,class_total_neighbors):

    n = G0.num_nodes #number of features

    M = np.zeros((n,n)) #initialize adjaceny matrix

    #Create a mapping from deature ID to node (for fast lookup)
    id_to_node = {node.id : node for node in G0.nodes}

    for (id1,id2,_) in G0.edges:
        f1 = id_to_node[id1]
        f2 = id_to_node[id2]

        spatial_distance = np.linalg.norm([f1.x - f2.x,f2.y - f2.y])
        temporal_distance = abs(f1.t - f2.t)        

        #compute ψs and ψt
        psi_s = gaussian(spatial_distance,mu_s,sigma_s)
        psi_t = gaussian(temporal_distance,mu_t,sigma_t)

        ci = f1.class_label
        cj = f2.class_label

        #compute uij again the stored counts
        uij = class_cooccurrence.get((ci,cj),0) / class_total_neighbors.get(ci,1) #avoid division by zero

        phi_ij = uij*psi_s*psi_t

        M[id1,id2] = phi_ij
        M[id2,id1] = phi_ij #since the graph is undirected (symmetric)

    return M,id_to_node

def build_degree_matrix_R(M):

    n = M.shape[0] #number of features

    R = np.zeros((n,n)) #initialzie degree matrix (diagonal)

    degrees = np.sum(M,axis=1) #sum over rows: sum of connection strengths

    for i in range(n):
        R[i,i] = degrees[i] #set only the diagonal elements
    
    return R

def build_laplacian_matrix_L(G0,class_cooccurrence,class_total_neighbors):

    M,_ = build_adjacency_matrix_M(G0,class_cooccurrence,class_total_neighbors)

    R = build_degree_matrix_R(M)

    L = R - M

    return L

def build_segment_laplacian(M_stip,stip_to_segment,num_segments):
    """
    Converts STIP-level Laplacian into segment-level Laplacian.
    """

    #Step 1: Initialize M_segment
    M_segment = np.zeros((num_segments,num_segments))

    #Step 3: Aggregate M_stip to M_segment
    for i in range(M_stip.shape[0]):
        seg_i = stip_to_segment[i]
        for j in range(M_stip.shape[1]):
            seg_j = stip_to_segment[j]
            M_segment[seg_i,seg_j] += M_stip[i,j]

    #Step 3: Degree and Laplacian
    R_segment = np.diag(np.sum(M_segment,axis=1))
    L_segment = R_segment - M_segment

    return L_segment,M_segment
