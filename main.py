from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id, udf
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import numpy as np
from pyspark.mllib.linalg import Vectors
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.sparse.linalg import svds


class MNISTKMeans:
    def __init__(self, file_path):
        self.spark = SparkSession.builder.appName("MNIST KMeans").getOrCreate()
        self.file_path = file_path
        self.data = None
        self.model = None
        self.transformed_data = None
        self.reduced_data = None

    def load_data(self):
        # Read CSV file
        df = self.spark.read.csv(self.file_path, header=False, inferSchema=True)
        # Assemble features
        feature_columns = [f"_c{i}" for i in range(1, 785)]
        self.assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        self.data = self.assembler.transform(df).select("features")
        # Add index column
        self.data = self.data.withColumn("index", monotonically_increasing_id().cast("integer"))

    def assign_weights(self):
        # Assign weights to special rows
        special_indices = {0, 1, 2, 3, 4, 7, 8, 11, 18, 61}

        # Define a UDF for weight assignment
        def assign_weight(index):
            return 100.0 if index in special_indices else 1.0

        assign_weight_udf = udf(assign_weight, DoubleType())

        # Apply weights
        self.data = self.data.withColumn("weight", assign_weight_udf(self.data["index"]))

    def fit_kmeans(self, k=10):
        kmeans = KMeans().setK(k).setSeed(1).setWeightCol("weight")
        self.model = kmeans.fit(self.data)
        self.transformed_data = self.model.transform(self.data)

    def compute_average_distances(self):
        # Extract cluster centers and broadcast them
        cluster_centers = self.model.clusterCenters()
        bc_centers = self.spark.sparkContext.broadcast(cluster_centers)

        distances = self.transformed_data.rdd.map(
            lambda row: (row.prediction, float(np.linalg.norm(row.features.toArray() - bc_centers.value[row.prediction])))
        )
        distances_df = distances.toDF(["cluster", "distance"])
        avg_distances = distances_df.groupBy("cluster").avg("distance").collect()
        return sorted(avg_distances, key=lambda x: x[0])

    def plot_distances(self, avg_distances):
        clusters, distances = zip(*[(x["cluster"], x["avg(distance)"]) for x in avg_distances])
        plt.bar(clusters, distances)
        plt.xlabel("Cluster")
        plt.ylabel("Average Distance to Centroid")
        plt.title("Average Distance from Data Points to Centroid per Cluster")
        plt.show()

    def plot_clusters(self):
        transformed_pandas = self.transformed_data.select("features", "prediction").toPandas()
        transformed_pandas["features"] = transformed_pandas["features"].apply(lambda x: x.toArray())

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(np.stack(transformed_pandas["features"].values))
        transformed_pandas["pca1"] = reduced_features[:, 0]
        transformed_pandas["pca2"] = reduced_features[:, 1]

        plt.figure(figsize=(14, 7))

        # Before clustering
        plt.subplot(1, 2, 1)
        plt.scatter(transformed_pandas["pca1"], transformed_pandas["pca2"], alpha=0.5)
        plt.title("Before Clustering")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")

        # After clustering with black centroids and number shapes
        plt.subplot(1, 2, 2)
        cluster_shapes = ['o', 's', 'D', 'P', 'X', '+', 'v', 'd', 'h', '8']  # Get 10 marker shapes

        for cluster in range(self.model.getK()):
            clustered_points = transformed_pandas[transformed_pandas["prediction"] == cluster]
            cluster_shape = cluster_shapes[cluster]  # Get shape for current cluster

            plt.scatter(clustered_points["pca1"], clustered_points["pca2"], alpha=0.5, label=f"Cluster {cluster}")

            # Get centroid for the current cluster
            centroid = self.model.clusterCenters()[cluster]
            centroid_pca = pca.transform(centroid.reshape(1, -1))  # Reshape for PCA

            # Plot the centroid as a black marker with the current shape
            plt.scatter(centroid_pca[0, 0], centroid_pca[0, 1], marker=cluster_shape, s=30, color="black", zorder=3, label=f"Centroid {cluster}")

        plt.title("After Clustering")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # Move legend outside plot

        plt.show()

    def reduce_dimensions(self):
        # Convert to Pandas DataFrame
        pandas_df = self.transformed_data.select("features", "prediction").toPandas()

        # Extract features and convert to numpy array
        features = np.array(pandas_df['features'].tolist())

        # Apply SVD to reduce dimensions to 3 components
        U, S, VT = svds(features, k=3)

        # The reduced features will be U * S
        reduced_features = np.dot(U, np.diag(S))

        # Add the reduced dimensions back into the Pandas DataFrame
        pandas_df['svd1'] = reduced_features[:, 0]
        pandas_df['svd2'] = reduced_features[:, 1]
        pandas_df['svd3'] = reduced_features[:, 2]
    
        # Convert back to Spark DataFrame and cache
        self.reduced_data = self.spark.createDataFrame(pandas_df[['svd1', 'svd2', 'svd3', 'prediction']]).cache()

    def plot_3d_clusters(self):
        # Sample data points
        sampled_data = self.reduced_data.sample(False, 100 / self.reduced_data.count())

        # Ensure that exactly 100 points are sampled
        sampled_data = sampled_data.limit(100)
        sampled_pandas = sampled_data.toPandas()

        # Plot 3D graph
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(sampled_pandas["svd1"], sampled_pandas["svd2"], sampled_pandas["svd3"], c=sampled_pandas["prediction"], cmap='viridis')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc='upper left', bbox_to_anchor=(0.8, 1.05))
        ax.add_artist(legend1)
        ax.set_xlabel("SVD Component 1")
        ax.set_ylabel("SVD Component 2")
        ax.set_zlabel("SVD Component 3")
        plt.title("3D Visualization of Clusters")
        plt.show()

    def run(self):
        self.load_data()
        #Task1
        self.assign_weights()
        self.fit_kmeans()
        avg_distances = self.compute_average_distances()
        self.plot_clusters()
        self.plot_distances(avg_distances)
        #Task2
        self.reduce_dimensions()
        self.plot_3d_clusters()

if __name__ == "__main__":
    mnist_kmeans = MNISTKMeans('/mnist_mini.csv')
    mnist_kmeans.run()
