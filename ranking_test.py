import numpy as np

emb = np.load("data/processed/image_embeddings.npy")

print(np.linalg.norm(emb[0]))
print(np.linalg.norm(emb[100]))