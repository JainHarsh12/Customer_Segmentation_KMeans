# customer_segmentation.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load and explore the dataset
def load_data():
    # Sample dataset; replace with actual dataset or load from a file
    data = {
        'CustomerID': [1, 2, 3, 4, 5],
        'Recency': [30, 200, 45, 10, 350],
        'Frequency': [20, 5, 15, 50, 2],
        'Monetary': [1000, 300, 500, 1200, 150]
    }
    df = pd.DataFrame(data)
    return df

# Preprocess data (remove unneeded columns, scale)
def preprocess_data(df):
    df_processed = df[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_processed)
    return df_scaled

# Elbow method to determine optimal number of clusters
def elbow_method(df_scaled):
    sse = []
    max_clusters = min(10, len(df_scaled))  # Ensure cluster count doesn’t exceed sample count
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title('Elbow Method to Determine Optimal K')
    plt.show()


# Fit K-means with a specified number of clusters
def fit_kmeans(df_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    return clusters, kmeans

# Analyze clusters and assign labels
def analyze_clusters(df, clusters):
    df['Cluster'] = clusters
    cluster_summary = df.groupby('Cluster').mean()
    print("\nCluster Summary:\n", cluster_summary)

    # Map cluster labels based on interpretation
    cluster_labels = {0: 'High Value', 1: 'Occasional Buyers', 2: 'Low Value'}
    df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
    print("\nLabeled Data:\n", df[['CustomerID', 'Cluster', 'Cluster_Label']])
    return df

# Visualize clusters
def visualize_clusters(df):
    plt.figure(figsize=(10, 6))
    for label in df['Cluster'].unique():
        plt.scatter(
            df[df['Cluster'] == label]['Frequency'],
            df[df['Cluster'] == label]['Monetary'],
            label=f"Cluster {label}"
        )
    plt.xlabel('Frequency')
    plt.ylabel('Monetary Value')
    plt.title('Customer Segments')
    plt.legend()
    plt.show()

# Main function to run all steps
if __name__ == "__main__":
    df = load_data()
    df_scaled = preprocess_data(df)
    elbow_method(df_scaled)  # Visualize the elbow method
    
    # Set optimal number of clusters (adjust based on elbow method)
    n_clusters = 3
    clusters, kmeans_model = fit_kmeans(df_scaled, n_clusters)

    # Calculate and print silhouette score for quality assessment
    silhouette_avg = silhouette_score(df_scaled, clusters)
    print(f"\nSilhouette Score: {silhouette_avg:.2f}")
    
    # Analyze and visualize the clusters
    df = analyze_clusters(df, clusters)
    visualize_clusters(df)
