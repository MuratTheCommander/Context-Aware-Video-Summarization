import numpy as np

class Graph:

    _counter = 0 #class-level counter for unique graph IDs

    def __init__(self,nodes,edges,name=None):
        Graph._counter += 1
        self.id = Graph._counter
        self.name = name if name else f"G{self.id}"
        self.nodes = nodes  #List of STIPFeature instances
        self.edges = edges  #List of (id1,id2,weight) tuples
        self.num_nodes = len(nodes)
        self.num_edges = len(edges)

    def __repr__(self):
        return f"<Graph {self.name}: {self.num_nodes} nodes, {self.num_edges} edges>"    

video_frame_width = 640  ###  use these values as default values
mu_s = 0.1*video_frame_width
sigma_s = 0.1*video_frame_width
mu_t = 90
sigma_t = 30

def compute_mu_sigma(stip_features,spatial_threshold,temporal_threshold):
    return   ### will be implemented later

def gaussian(distance,mu,sigma):
    return  (1.0 / (np.sqrt(2.0*np.pi) * sigma)) * np.exp(-((distance - mu) ** 2) / (2 * (sigma ** 2)))


def build_edges(stip_features,spatial_threshold,temporal_threshold,class_cooccurrence,class_total_neighbors):

    edges = []

    n = len(stip_features)

    for i in range(n):
        for j in range(i+1,n):
            f1 = stip_features[i]
            f2 = stip_features[j]

            spatial_distance = np.linalg.norm([f1.x - f2.x,f1.y - f2.y])
            temporal_distance = abs(f1.t - f2.t)

            if spatial_distance < spatial_threshold or temporal_distance < temporal_threshold :

                #compute ψs and ψt
                psi_s = gaussian(spatial_distance,mu_s,sigma_s)
                psi_t = gaussian(temporal_distance,mu_t,sigma_t)

                #update class occurence counts for uij
                ci = f1.class_label
                cj = f2.class_label

                #update co-occurence counts
                class_cooccurrence[(ci,cj)] = class_cooccurrence.get((ci,cj),0) + 1
                class_total_neighbors[ci] = class_total_neighbors.get(ci,0) + 1

                #compute uij
                uij = class_cooccurrence[(ci,cj)] / class_total_neighbors[ci] if class_total_neighbors[ci] > 0 else 0.0

                #final edge weight
                edge_weight = uij*psi_s*psi_t

                #save the edge with real feature ids
                edges.append((f1.id,f2.id,edge_weight))


    return edges,class_cooccurrence,class_total_neighbors

def build_Graph(stip_features,edges,name):
    G = Graph(stip_features,edges,name)
    return G



