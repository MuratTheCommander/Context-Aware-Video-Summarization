import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment

def class_histogram_vector(G):
    labels = [node.class_label for node in G.nodes]

    counter = Counter(labels)

    all_classes = list(set(counter.keys()))

    vec = np.array([counter[c] for c in all_classes])

    return vec,all_classes
                   
def compute_graph_similarity(G1,G2,alpha=0.5,beta=0.5):
    """
    Computes fast similarity score ∈ [0,1] between two graphs.
    alpha/beta control weighting of histogram vs structural features.
    """

    # 1.Class label histogram similarity cosine
    h1, classes1 = class_histogram_vector(G1)
    h2, classes2 = class_histogram_vector(G2)

    # Align classes
    all_classes = list(set(classes1 + classes2))

    v1 = np.array([Counter([n.class_label for n in G1.nodes]).get(c,0) for c in all_classes])

    v2 = np.array([Counter([n.class_label for n in G2.nodes]).get(c,0) for c in all_classes])

    hist_sim = np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e -8)


    # 2.Structural similarity (node + edge stats)

    node_diff = abs(len(G1.nodes) - len(G2.nodes)) / max(len(G1.nodes),len(G2.nodes),1)

    edge_diff = abs(len(G1.edges) - len(G2.edges)) / max(len(G1.edges),len(G2.edges),1)

    weight1 = [w for (_,_,w) in G1.edges]

    weight2 = [w for (_,_,w) in G2.edges]

    mean_w1 = np.mean(weight1) if weight1 else 0

    mean_w2 = np.mean(weight2) if weight2 else 0

    weight_diff = np.abs(mean_w1 - mean_w2) / max(mean_w1,mean_w2,1e-6)

    struct_score = 1.0 - 0.5 * (node_diff + edge_diff + weight_diff)

    #Weighted combination
    total_score = alpha*hist_sim + beta*struct_score

    return total_score

def node_cost(f1, f2, use_class_label=True, use_descriptor=True, descriptor_weight=1.0):

    cost = 0.0

    # Semantic match based on visual word label
    if use_class_label:
        if f1.class_label != f2.class_label:
            cost += 100 # Peanlize different labels heavily

    #Visual similarity (HOG+HOF descriptor L2 distance)
    if use_descriptor:
        if f1.descriptor is not None and f2.descriptor is not None:
            cost +=  descriptor_weight * np.linalg.norm(f1.descriptor - f2.descriptor)

    return cost

def construct_H_matrix(g_test,g_candidate):
    
    n = len(g_test.nodes)

    m = len(g_candidate.nodes)

    H = np.zeros((n,m))

    for i,f_test in enumerate(g_test.nodes):
        for j,f_candidate in enumerate(g_candidate.nodes):

            H[i,j] = node_cost(f_test,f_candidate)

    return H

def compute_h_score(H):
    
    row_ind,col_ind = linear_sum_assignment(H) # Finds optimal assignment

    total_cost = H[row_ind,col_ind].sum()
    
    average_cost = total_cost / len(row_ind) # Normalize
    
    return average_cost

def match_graphs(g_test,g_candidate):

    H = construct_H_matrix(g_test,g_candidate)

    h_score = compute_h_score(H)

    return h_score

def sv(node_a,node_b,class_label_match=True,descriptor_threshold=0.3):
    if class_label_match:
        return node_a.class_label == node_b.class_label

    if node_a.descriptor is not None and node_b.descriptor is not None:
        dist = np.linalg.norm(node_a.descriptor - node_b.descriptor) 
        return dist < descriptor_threshold
    
    return False

def se(weight_a,weight_b,tolerance=0.5):
    return np.abs(weight_a - weight_b) <= tolerance

def merge_graphs(g_test,g_ref):
    updated_edges = []

    #Build lookup: id -> STIPFeature for both graphs
    test_nodes = {n.id: n for n in g_test.nodes}
    ref_nodes = {n.id: n for n in g_ref.nodes}

    for id1_t,id2_t,weight_t in g_test.edges:
        match_found = False
        for id1_r,id2_r,weight_r in g_ref.edges:
            node1_t,node2_t = test_nodes.get(id1_t),test_nodes.get(id2_t)
            node1_r,node2_r = ref_nodes.get(id1_r),ref_nodes.get(id2_r)

            if (sv(node1_t,node1_r) and sv(node2_t,node2_r)): 
                match_found = True
                if not se(weight_t,weight_r):
                    #update the edge weight
                    updated_edges.append((id1_r,id2_r,weight_t))
                break

        #if no match at all,add test edge as new
        if not match_found:
            updated_edges.append((id1_t,id2_t,weight_t))

    #Apply updates to the reference graph
    for id1,id2, new_weight in updated_edges:
        #Remove any existing edge between id1-id2
        g_ref.edges = [e for e in g_ref.edges if not ((e[0] == id1 and e[1] == id2) or (e[0] == id2 and e[1] == id1))]
        g_ref.edges.append((id1,id2,new_weight))

    print(f"Graph {g_ref.name} updated with info from {g_test.name}")

    

    



