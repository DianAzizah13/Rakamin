#!/usr/bin/env python
# coding: utf-8

# The purpose of creating this machine learning model is to be able to cluster similar customers.

# # Load Dataset

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# scalling
from sklearn.preprocessing import StandardScaler
# modelling
from sklearn.cluster import KMeans
#silhoute
import sklearn.cluster as cluster
import sklearn.metrics as metrics


# In[3]:


import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

# Or, only disable specific warnings based on category
# Example: Disabling DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[4]:


df = pd.read_csv("df_kalbe.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df = df.drop(["storeid.1", "productid.1", "price.1", "customerid.1"], axis=1)


# In[8]:


df.info()


# In[9]:


df.shape


# In[10]:


df.duplicated().info()


# In[11]:


df.isnull().info()


# # Data Aggregation

# In[12]:


aggregated_data = df.groupby('customerid').agg({
    'transactionid': 'count',
    'qty': 'sum',
    'totalamount': 'sum'
}).reset_index()


# In[13]:


aggregated_data.rename(columns={'transactionid' : "TotalTransaction", 'qty' : 'TotalQuantity', 'totalamount' : "TotalAmount"}, inplace=True)


# In[14]:


aggregated_data.info()


# In[15]:


aggregated_data.describe()


# In[16]:


sns.pairplot(data = aggregated_data)


# In[17]:


# Menghitung korelasi antara variabel
correlation_matrix = aggregated_data[['TotalTransaction','TotalQuantity', 'TotalAmount']].corr()

# Membuat heatmap untuk visualisasi korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()


# In[18]:


# Scaling
sc = StandardScaler()
dfoutlier_std = sc.fit_transform(aggregated_data[['TotalTransaction','TotalQuantity','TotalAmount']].astype(float))

new_dfoutlier_std = pd.DataFrame(data = dfoutlier_std, columns = ['TotalTransaction','TotalQuantity','TotalAmount'])
     


# In[19]:


new_dfoutlier_std.head()


# In[20]:


sns.pairplot(new_dfoutlier_std)


# In[21]:


# Elbow Method
# declare Within-Cluster Sum of Squares (wcss)
wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init='k-means++', random_state = 42)
  kmeans.fit(new_dfoutlier_std)
  wcss.append(kmeans.inertia_)
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# From elbow we can choose 3 or 4 cluster for K

# In[22]:


# Silhoute Method
for i in range(2,13):
    labels=cluster.KMeans(n_clusters=i,init="k-means++",random_state=42).fit(new_dfoutlier_std).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score(new_dfoutlier_std,labels,metric="euclidean",random_state=42)))
     


# In[38]:


# k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42).fit(new_dfoutlier_std)
labels = kmeans.labels_
new_dfoutlier_std['label_kmeans'] = labels


# In[39]:


new_dfoutlier_std.head()


# In[40]:


colors_cluster=['yellow','green', 'brown']
label_cluster=['cluster 0', 'cluster 1', "cluster 2"]

# PLOTTING
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for points in clusters
for cluster_id in range(3):
    cluster_data = new_dfoutlier_std[new_dfoutlier_std["label_kmeans"] == cluster_id]
    ax.scatter(cluster_data["TotalTransaction"], cluster_data["TotalQuantity"], cluster_data["TotalAmount"],
               c=colors_cluster[cluster_id], s=30, edgecolor='brown', label=label_cluster[cluster_id])

# Scatter plot for cluster centers (red)
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=200, label='Cluster Centers')

# Set labels and legend
ax.set_xlabel('TotalTransaction')
ax.set_ylabel('TotalQuantity')
ax.set_zlabel('TotalAmount')
ax.legend()

plt.show()


# In[41]:


# copy label cluster to origin data
df_customer_clustering = aggregated_data.copy()
df_customer_clustering['cluster'] = kmeans.labels_


# In[42]:


df_customer_clustering.head()


# In[43]:


# Calculate the average metrics for each cluster, including customer count, mean TotalTransaction, mean TotalQuantity, and mean TotalAmount.
df_customer_clustering.groupby('cluster').agg({'customerid':'count',
                                               'TotalTransaction':'mean',
                                               'TotalQuantity':'mean',
                                               'TotalAmount':'mean'
                                               }).sort_values(by='TotalAmount').reset_index()


# Cluster 2 has the highest number of customers.
# 
# Cluster 0 (Moderate Shopper): This indicates that customers in this cluster tend to make purchases of regular quantity and value.
# 
# Cluster 1 (High Value Shopper): This indicates that customers in this cluster tend to make purchases with high monetary value.
# 
# Cluster 2 (Balanced Shopper): This indicates that customers in this cluster exhibit balanced purchasing behavior in terms of quantity and value.

# In[30]:


df_customer_clustering.to_csv('df_customer_clustering.csv',index=False)
     


# In[31]:


# Set style
sns.set(style="whitegrid")

# Define colors for clusters
cluster_colors = ['red','yellow','blue']

# Create scatter plots for each feature pair
feature_pairs = [('TotalTransaction', 'TotalQuantity'), ('TotalTransaction', 'TotalAmount'), ('TotalQuantity', 'TotalAmount')]

for pair in feature_pairs:
    plt.figure(figsize=(10, 6))
    for cluster_num in range(len(cluster_colors)):
        cluster_data = new_dfoutlier_std[new_dfoutlier_std.label_kmeans == cluster_num]
        plt.scatter(cluster_data[pair[0]], cluster_data[pair[1]], color=cluster_colors[cluster_num], label=f'Cluster {cluster_num}')
    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    plt.title(f'Scatter Plot of {pair[0]} vs {pair[1]}')
    plt.legend()
    plt.show()


# Cluster 0 (Medium Spender):
# 
# Marketing Strategy: Encourage them to shop more frequently to increase their transaction value. Send marketing product notifications via email.
# Promotions: Special promotions and discounts to motivate this cluster to shop more frequently.
# 
# Cluster 2 (Balanced Shoppers):
# 
# Marketing Strategy: Customers in this cluster have a medium transaction value, which indicates that they are quite active. Focus on maintaining their activity level and building further engagement.
# Promotion: Offer exclusive discounts or special deals that are only available to customers in this cluster as a token of appreciation.
# 
# Cluster 1 (High Value Shoppers):
# 
# Marketing Strategy: Focus on maintaining strong relationships with customers in this cluster, as they contribute significantly to the business.
# Promotion: Introduce a premium loyalty program that offers additional benefits to customers in this cluster. This could include exclusive discounts, early access to new products, or luxury gifts.

# In[ ]:




