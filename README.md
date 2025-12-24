# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Collection
- Gather customer data (features such as age, annual income, spending score, purchase frequency, etc.).
- No target variable is required since clustering is unsupervised.
Step 2: Data Preprocessing
- Handle missing values (impute or drop).
- Normalize/scale numerical features (important for K-Means since it uses distance-based calculations).
- Encode categorical variables if present.
Step 3: Choose Number of Clusters (k)
- Use the Elbow Method or Silhouette Score to determine the optimal number of clusters.
Step 4: Apply K-Means Algorithm
- Initialize k cluster centroids randomly.
- Assign each data point to the nearest centroid (based on Euclidean distance).
- Update centroids by calculating the mean of all points in each cluster.
- Repeat assignment and update steps until centroids stabilize or maximum iterations are reached.
Step 5: Prediction (Cluster Assignment)
- Each customer is assigned to a cluster (e.g., Cluster 1 = high spenders, Cluster 2 = budget-conscious, etc.).
Step 6: Evaluation
- Visualize clusters using scatter plots (e.g., income vs. spending score).
- Interpret clusters for business insights (e.g., marketing strategies for each segment).


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.

*/
```
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10],
    'Gender': ['Male','Female','Female','Male','Female','Male','Male','Female','Female','Male'],
    'Age': [19,21,20,23,31,22,35,30,25,28],
    'Annual Income (k$)': [15,16,17,18,19,20,21,22,23,24],
    'Spending Score (1-100)': [39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Select features for clustering
# ------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ------------------------------
# Step 3: Apply K-Means (choose clusters, e.g., 3)
# ------------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)  # Automatically fits and assigns clusters

# ------------------------------
# Step 4: Visualize clusters
# ------------------------------
plt.figure(figsize=(8,6))
for i in range(3):
    plt.scatter(X[df['Cluster']==i]['Annual Income (k$)'],
                X[df['Cluster']==i]['Spending Score (1-100)'],
                label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=200, c='yellow', label='Centroids', marker='X')

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# ------------------------------
# Step 5: Show dataset with clusters
# ------------------------------
print(df)


## Output:
K Means Clustering for Customer Segmentation

<img width="864" height="719" alt="image" src="https://github.com/user-attachments/assets/36f4bb10-bbdd-40e9-9059-48683d8bde74" />

<img width="785" height="582" alt="image" src="https://github.com/user-attachments/assets/d36d4c0c-93ff-498b-950b-ba9d9ef27405" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
