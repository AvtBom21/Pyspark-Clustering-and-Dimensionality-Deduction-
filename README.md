Clustering using KMeans:
KMeans is a popular clustering algorithm used to divide a data set into K different clusters. The goal of KMeans is to minimize the sum of squared distances between data points and their cluster centers.

How it works:

Initialization: Choose K initial cluster centers (randomly or use a method like KMeans++ to choose better initialization centers). Cluster assignment: Assign each data point to the cluster with the closest center.
Update: Recalculate the cluster center by taking the average of the points in each cluster.
Repeat: Repeat the two steps of cluster assignment and update until the cluster center does not change (or changes very little).
In the code, MNIST data (handwritten digit image data) is used to cluster into 10 clusters (corresponding to 10 digits from 0 to 9). The code also assigns weights to particular data points, helping the algorithm pay more attention to these points.
Advantages of K-means:

Simplicity and ease of understanding
Time efficiency
Scalability
Effective with spherical clusters
Disadvantages of K-means:

Difficulty in choosing the number of clusters (k)
Sensitivity to initial values
Limitation to spherical clusters
Does not perform well with heterogeneous data
Susceptible to outliers and noise
Hard clustering
TruncatedSVD:
Truncated SVD (Singular Value Decomposition) is a data dimensionality reduction technique often used in the field of machine learning and big data processing, especially for sparse data. This is a variation of the standard SVD, optimized to handle large matrices without needing to compute the entire covariance matrix.
How it works:
Given a data matrix A of size m×n, SVD decomposes A into three matrices: image.png
Truncated SVD: Truncated SVD retains only k principal components, where k is less than or equal to the dimensionality of the original data. This helps reduce the size of the data while preserving the most important information. Matrix A is approximated by: image.png
Advantages of SVD:
Effective dimensionality reduction
Noise reduction
Optimal low-rank approximation
Handling of correlated features
Applications in diverse fields
Disadvantages of SVD:
Computationally expensive
Interpretability issues
Sensitivity to data scaling
Data requirement
Storage requirements
