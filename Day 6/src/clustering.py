import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#load the data
df = pd.read_csv("data/store_Customers.csv")

print(df.head())

#basic info about the data
print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

#statistical summary
print(df.describe())

df = df.dropna()

#drop the 'CustomerID' column as it is not useful for clustering
df = df.drop("CustomerID", axis=1)

#convert gender to numeric
df["Gender"] = df["Gender"].map({
    "M": 0,
    "F": 1
})

print(df.head())

#scale the features
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

#verify the scaled data

print(scaled_data[:5])


#apply PCA to reduce dimensionality to 2 components for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)


pca_full = PCA()
pca_full.fit(scaled_data)

explained_variance = np.cumsum(pca_full.explained_variance_ratio_)


#plot scree plot to determine the number of components to keep
plt.figure(figsize=(8,5))
plt.plot(explained_variance, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Scree Plot")
plt.grid()
plt.show()

#run kmeans for different values of k and plot the results
from sklearn.cluster import KMeans

inertia = []
K = range(2, 9)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_data)
    inertia.append(kmeans.inertia_)

#elbow plot to determine the optimal number of clusters
plt.figure(figsize=(8,5))
plt.plot(K, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid()
plt.show()


#compute silhouette scores for different values of k
from sklearn.metrics import silhouette_score

silhouette_scores = []
K = range(2, 9)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_data)
    
    score = silhouette_score(pca_data, labels)
    silhouette_scores.append(score)

#plot silhouette scores
plt.figure(figsize=(8,5))
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal K")
plt.grid()
plt.show()

#train kmeans with the optimal number of clusters (k=5 based on elbow and silhouette analysis)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(pca_data)

#add labels to the original dataframe
df["Cluster"] = labels
print(df.head())

#visualize the clusters in the PCA space
plt.figure(figsize=(8,6))

plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=labels,
    cmap='viridis',
    s=50
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Segments (KMeans + PCA)")
plt.grid()
plt.show()