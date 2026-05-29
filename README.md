# KMeans-Iris-Clustering-Project

# Unsupervised Learning with KMeans

This is a beginner machine learning project applying KMeans clustering to the classic Iris dataset with no labels given to the algorithm.

---

## What is Unsupervised Learning?

In unsupervised learning, the model is given data **without labels**. It has to find patterns, structure, or groupings entirely on its own. There is no "right answer" provided during training, the algorithm discovers hidden structure by itself.

This project uses **KMeans Clustering**, one of the most popular unsupervised learning algorithms, to group 150 iris flowers into clusters based purely on their physical measurements.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Iris Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

The dataset contains 150 rows and 5 columns:

| Column | Description |
|---|---|
| sepal_length | Length of the sepal (cm) |
| sepal_width | Width of the sepal (cm) |
| petal_length | Length of the petal (cm) |
| petal_width | Width of the petal (cm) |
| species | Iris species (setosa, versicolor, virginica) |

---

## Project Pipeline

### Stage 1 - Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

Three essential libraries are imported before anything else:

| Library | Purpose |
|---|---|
| `pandas` | Loading and manipulating the dataset as a DataFrame |
| `numpy` | Converting data into arrays for the machine learning algorithm |
| `matplotlib` | Visualising the clusters and elbow curve |

---

### Stage 2 - Loading and Exploring the Dataset

```python
data = pd.read_csv('/Users/zee/Downloads/iris.csv')
data.info()
```

Loads the Iris CSV file into a pandas DataFrame called `data`. `.info()` gives a quick overview of the dataset, showing the number of rows, column names, data types, and whether any values are missing. This is an important first step to understand what you are working with before any processing begins.

```python
data.head()
```

Displays the first 5 rows of the dataset. This gives a visual preview of the data, confirming the columns loaded correctly and showing what the values actually look like.

---

### Stage 3 - Preprocessing

```python
x = data.drop('species', axis=1)
x
```

The `species` column is the label. Since unsupervised learning requires no labels, it is removed. `axis=1` tells pandas to drop a column (not a row). The result is stored in `x`, leaving the original `data` untouched.

---

### Stage 4 - Converting to a NumPy Array

```python
x = x.values
x
```

The pandas DataFrame is converted to a NumPy array using `.values`. This strips away column names and index labels, leaving only raw numbers, which is what scikit-learn's algorithms expect as input.

---

### Stage 5 - Building and Training the KMeans Model

```python
from sklearn.cluster import KMeans

c3 = KMeans(n_clusters=3)
c3.fit(x)
```

- `KMeans` is imported from scikit-learn's `cluster` module
- `n_clusters=3` tells the algorithm to find 3 groups
- `.fit(x)` the algorithm groups the 150 data points into 3 clusters by finding similar data points

---

### Stage 6 - Viewing the Cluster Centers

```python
c3.cluster_centers_
```
---

### Stage 7 - Viewing the Cluster Labels

```python
c3.labels_
```
---

### Stage 8 - Checking Inertia

```python
c3.inertia_
```

Inertia measures the sum of distances between each data point and the center of its cluster. Lower inertia means tighter, more compact clusters. It is used to evaluate and compare model quality.

---

### Stage 9 - Visualising the Clusters

```python
import matplotlib.pyplot as plt

plt.scatter(x[:,2], x[:,3], c=c3.labels_)
plt.title('Iris Clusters by Petal Measurements')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()
```

Plots petal length vs petal width, colored by cluster assignment. Petal measurements are used because they show the clearest separation between the 3 groups.

---

### Stage 10 - Visualising Clusters with Centroids

```python
plt.scatter(x[:,2], x[:,3], c=c3.labels_)
plt.scatter(c3.cluster_centers_[:,2], c3.cluster_centers_[:,3], color='red')
plt.title('Iris Clusters with Centroids')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()
```
---

### Stage 11 - Calculating Inertia for Multiple K Values

```python
inertias = []
for K in range(1, 11):
    cK = KMeans(n_clusters=K)
    cK.fit(x)
    inertias.append(cK.inertia_)
```
---

### Stage 12 - Plotting the Elbow Curve

```python
plt.plot(range(1, 11), inertias)
plt.title('Elbow Method - Optimal Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()
```
---

### Stage 13 - Comparing Clusters to Real Species Labels

```python
import pandas as pd

data['cluster'] = c3.labels_
print(data.groupby(['species', 'cluster']).size())
```
---

## Results

```
species     cluster
setosa      0          50
versicolor  1           3
            2          47
virginica   1          36
            2          14
```

| Species | Correct | Misclassified | Accuracy |
|---|---|---|---|
| Setosa | 50 | 0 | 100% |
| Versicolor | 47 | 3 | 94% |
| Virginica | 36 | 14 | 72% |

**Key findings:**
- Setosa was perfectly separated- its small petal measurements are distinctly different from the other two species
- Versicolor was mostly correct with only 3 misclassifications
- Virginica had more overlap with versicolor, which is a well-known characteristic of this dataset-the two species share similar petal measurements

The algorithm achieved these results **without ever being told the species labels** it discovered the groupings purely from patterns in the numbers.

---

## What I Learned

- Unsupervised learning finds hidden patterns in data without any labels
- KMeans groups data by minimising the distance between points and their cluster centers
- Inertia measures how compact the clusters are; lower is better
- The Elbow Method helps identify the optimal number of clusters
- Even without labels, KMeans closely matched the real species groupings,proving it found genuine structure in the data
- Versicolor and virginica naturally overlap in their measurements, which is why they are harder to separate than setosa

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

---

## How to Run

```bash
git clone https://github.com/EzinneUfondu/kmeans-iris-clustering
cd kmeans-iris-clustering
pip install -r requirements.txt
jupyter notebook kmeans_clustering.ipynb
```
