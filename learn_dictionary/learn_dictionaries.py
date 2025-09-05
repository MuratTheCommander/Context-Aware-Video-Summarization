from sklearn.cluster import KMeans
import numpy as np
    

def initialize_Df(X):
    X = X.T #from (136,num_segments) to (num_segments,136)  
    
    kmeans = KMeans(n_clusters=40,random_state=20)
    kmeans.fit(X)

    Df = kmeans.cluster_centers_.T #from (num_segments,136) to (136,num_segments)

    print(f"X shape: ", X.shape)
    print(f"Df shape: ", Df.shape)
    print("First atom: ", Df[:,0])
    print("Last atom: ", Df[:,-1])

    return Df

def optimize_Df(X,B,Df,epsilon = 1e-6):

    for j in range(B.shape[0]):
        bj = B[j,:] #All coefficients for atom j

        if np.linalg.norm(bj) < epsilon:
            continue #skip if  atom atom is unused

        #Residual = data -full reconstruction + this atom's contrbution        
        R = X - Df @ B + np.outer(Df[:,j],bj)

        # Least-squares update for atom j
        dj = R @ bj.T / (np.linalg.norm(bj)**2)

        #Normalize to unit length
        Df[:,j] = dj / max(np.linalg.norm(dj),epsilon)
    return Df








