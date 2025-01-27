import numpy as np
import matplotlib.pyplot as plt
import sys

def initWFCMCenters(X, weights, K):
    """
    Perform Weighted fuzzy C-means++ initialization.
    
    X: Data points (N x D), where N is the number of points and D is the feature dimension.
    weights: Weights of data points (N,)
    K: Number of cluster centers to initialize
    
    Returns: 
        centers: K initial cluster centers

    Inspiration: D.Arthur and S. Vassilvitskii. 2007. k-means++: The Advantages of Careful Seeding. Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) (2007), 1027-1035.
    """
    N = X.shape[0]
    
    # Step 1: Compute the total sum of the weights
    W = np.sum(weights)
    
    # Step 2: Initialize the selection probabilities based on weights
    probabilities = weights / W  # Initial probabilities based on the weight of each point
    
    # Step 3: Select the first center randomly based on weight
    centers = []
    first_center_idx = np.random.choice(N, p=probabilities)  # Choose the first center based on weights
    centers.append(X[first_center_idx])
    
    # Step 4: Select the remaining K-1 centers
    for _ in range(1, K):
        # Step 4a: Compute the squared distances of each point to the nearest selected center
        dist_sq = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centers), axis=2) ** 2, axis=1)
        
        # Step 4b: Update the probabilities by considering both distance and weight
        weighted_dist_sq = dist_sq * weights  # Multiply the squared distances by the weights
        probabilities = weighted_dist_sq / np.sum(weighted_dist_sq)
        
        # Step 4c: Select the next center based on the updated probabilities
        next_center_idx = np.random.choice(N, p=probabilities)
        centers.append(X[next_center_idx])
    
    return np.array(centers)




def WFCM(X, c,  max_iter=300, tol=1e-4,m_i = 4, m = 2,weights=None, loose_index = None):
    """
    Weighted Fuzzy C-Means (WFCM) Algorithm

    Parameters:
    - X: Data set, shape (n_samples, n_features)
    - c: Number of clusters
    - max_iter: Maximum number of iterations
    - tol: Tolerance for stopping criterion
    - m: Fuzziness parameter, typically 2
    - m_i: Max influencer
    - weights: Weights of the data points, shape (n_samples,), if None, all weights are 1

    Returns:
    - U: Membership matrix, shape (n_samples, c)
    - centers: The center of each cluster, shape (c, n_features)

    Inspiration: James C. Bezdek, Robert Ehrlich, and William Full. 1984. FCM: The fuzzy c-means clustering algorithm. Computers & Geosciences 10, 2 (1984), 191-203. https://doi.org/10.1016/0098-3004(84)90020-7
    """
    def max_influencer_norm(W:np.array, m_i):
        ans = np.zeros_like(W)
        for i in range(W.shape[0]):
            row = W[i]
            top_indices = np.argsort(row)[-m_i:]
            ans[i,top_indices] = row[top_indices]
        ans = ans/ans.sum(axis=1, keepdims=True)
        return ans
    
    n_samples, n_features = X.shape
    
    # If no weights are provided, assume uniform weights
    if weights is None:
        weights = np.ones(n_samples)
    
    # Initialize cluster centers randomly
    # centers = np.random.rand(c, n_features)
    centers = initWFCMCenters(X, weights, c)


    # Initialize membership matrix randomly
    U = np.random.rand(n_samples, c)
    U = max_influencer_norm(U,m_i)



    # Iterative process of Weighted Fuzzy C-Means clustering
    for iteration in range(max_iter):
        # Store the previous cluster centers
        old_centers = centers.copy()
        
        # Calculate the weighted distance between points and cluster centers
        dist = np.linalg.norm(X[:, np.newaxis] - centers,axis=2)

        # Update membership matrix
        # Calculate new membership based on weighted distance
        dist_weighted = dist ** 2 / (weights[:,np.newaxis] ** (2 / (m - 1)))
        U= 1 / (dist_weighted + 1e-10)

        # Normalize the membership matrix
        U = max_influencer_norm(U,m_i)

        # Update the cluster centers considering weights
        numerator = np.sum(((U ** m) * weights[:, np.newaxis])[:,:, np.newaxis] * X[:,np.newaxis], axis=0)
        denominator = np.sum((U ** m) * weights[:, np.newaxis], axis=0)
        centers = numerator / (denominator[:,np.newaxis] + 1e-10)

        # Check for convergence (when the centers don't change much)
        center_diff = np.linalg.norm(centers - old_centers)  # Compare with previous centers
        if center_diff < tol:
            print(f'Converged at iteration {iteration}')
            break
   
    print(f'WFCM Complete, rencent difference:{center_diff}')
    return centers, U

if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(500, 3) + np.array([1, 1,1]),  # Cluster 1
        np.random.randn(500, 3) + np.array([2, 2,2]) # Cluster 2
    ])

    X = np.vstack([X,[7,7,7]])
    X = X * 0.1

    # Set weights (example: some points in Cluster 1 are given higher weights)
    weights = np.ones(X.shape[0])
    weights[:5] = 3  # The first 5 points in Cluster 1 are more important

    # Number of clusters
    c = 2
    m = 2  # Fuzziness parameter

    # Run the Weighted Fuzzy C-Means clustering
    centers , U = WFCM(X, c, m, weights=weights)
    def renderWeights(vertices:np.array,
                        bone_pos: np.array,
                        weights: np.array,
                        faces=None
                        ):
        import open3d as o3d
        # distrib
        base_colors = np.array([
        [0.9, 0.9, 0.7],  # (Light Yellow)
        [0.7, 0.9, 0.7],  # (Light Green)
        [0.7, 0.9, 1],    # (Light Blue)
        [0.9, 0.8, 0.9],  # (Light Purple)
        [1, 0.8, 0.6],    # (Peach Orange)
        # [0.8, 0.7, 0.5],  # (Sand Brown)
        [0.6, 0.8, 0.8],  # (Light Cyan)
        [0.8, 0.8, 0.8],  # (Light Gray)
        ])
        base_colors2 = np.array([
        [1, 0, 0],    # Red
        [0, 1, 0],    # Green
        [0, 0, 1],    # Blue
        [1, 1, 0],    # Yellow
        [1, 0.647, 0], # Orange
        [0.5, 0, 0.5], # Purple
        [0, 1, 1],    # Cyan
        # [0.6, 0.6, 0.6] # Light Gray
        ])

        bone_colors = np.array([base_colors2[i%7] for i in range(bone_pos.shape[0])])
        vertices_colors = np.matmul(weights, bone_colors)

        if faces is not None:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertices_colors)
        else:
            mesh = o3d.geometry.PointCloud()
            mesh.points = o3d.utility.Vector3dVector(vertices)
            mesh.colors = o3d.utility.Vector3dVector(vertices_colors)


        bones = []
        for i, position in enumerate(bone_pos):
            b = o3d.geometry.TriangleMesh.create_box(width=0.03, height=0.03, depth=0.03)
            b.translate(position)
            b.paint_uniform_color(bone_colors[i])  # 设置该立方体的颜色
            bones.append(b)

        o3d.visualization.draw_geometries([mesh] + bones)

        pass

    print(U)
    print(centers)
    # Visualize the clusteri ng results
    renderWeights(X,  centers,U)

