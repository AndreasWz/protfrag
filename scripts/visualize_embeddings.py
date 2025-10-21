# scripts/visualize_embeddings.py
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
X = []
labels = []
for _, r in df.iterrows():
    emb = np.load(r['embedding_path'] if r['embedding_path'] else f"data/embeddings/{r['id']}.npy")
    X.append(emb.mean(axis=0))
    labels.append(int(r['is_fragment']))
X = np.vstack(X)
proj = PCA(n_components=50).fit_transform(X)
u = umap.UMAP(n_components=2).fit_transform(proj)
plt.scatter(u[:,0], u[:,1], c=labels, s=5, cmap="coolwarm")
plt.savefig("out/umap_frag_vs_complete.png", dpi=200)
