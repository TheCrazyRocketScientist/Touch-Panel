from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Get raw data and labels

from label_data import get_train_data
X_train,y_train = data,labels = get_train_data(window_size=0.6,overlap=0.2)


X_train = X_train[y_train!=0]
y_train = y_train[y_train!=0]


# Flatten to shape (N, 42)
X_flat = X_train.reshape(X_train.shape[0], -1)

from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X_flat)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y_train, cmap='tab10', s=20)
ax.set_title('3D PCA of Accelerometer Windows')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()



# ----- t-SNE -----
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_flat)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_train, palette='tab10')
plt.title("t-SNE of Raw Accelerometer Windows")
plt.show()
