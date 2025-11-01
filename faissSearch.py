import faiss
import numpy as np
import time

start = time.perf_counter()

# Each vector has 3 dimensions
dim = 3  

# Define your 5 vectors
a = np.array([0.1, 0.3, 0.4])
b = np.array([-0.1, -0.3, -0.4])
c = np.array([7, 8, 9])
d_vec = np.array([100, 1.2, -678])
e = np.array([-2, 1.4, 20])


q = np.array ([6.9, 0, 100])
# Stack all into a 2D array of shape (5, 3)
xb = np.stack((a, b, c, d_vec, e)).astype('float32')

# Query vector(s) — also needs shape (n, dim)
xq = np.array([q]).astype('float32')

# Create index (L2 distance)
index = faiss.IndexFlatL2(dim)

# Add your database vectors to the index
index.add(xb) 

# Search for the 1 nearest neighbor of xq
D, I = index.search(xq, k=1)

print("Distances: ", D[0][0])
print("Indices:", I[0][0])


end = time.perf_counter()
print(f"Runtime: {end - start:.6f} seconds")