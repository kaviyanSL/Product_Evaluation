import logging
from sklearn.cluster import KMeans

class ClusteringService():
    def __init__(self, reviews, vectorize_reviews, n_clusters = 3):
        self.reviews = reviews
        self.vectorize_reviews = vectorize_reviews
        self.n_clusters = n_clusters

    def cluster_reviews(self):
        vectorizer_data = self.vectorize_reviews
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(vectorizer_data)
        clusters = kmeans.labels_
        return clusters

    def get_clustered_reviews(self):
        clusters = self.cluster_reviews()
        clusters = clusters.tolist()
        clustered_reviews = {}
        for i, cluster in enumerate(clusters):
            cluster = int(cluster)
            if cluster not in clustered_reviews:
                clustered_reviews[cluster] = []
            clustered_reviews[cluster].append(self.reviews[i])
        logging.info(f"Clustered reviews is done")

        vectorized_reviews = {}    
        for i, cluster in enumerate(clusters):
            cluster = int(cluster)
            if cluster not in vectorized_reviews:
                vectorized_reviews[cluster] = []
            vectorized_reviews[cluster].append(self.vectorize_reviews[i])
        logging.info(f"Clustered Vectorized reviews is done")
        
        return clustered_reviews, vectorized_reviews