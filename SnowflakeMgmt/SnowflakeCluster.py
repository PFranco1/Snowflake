import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pandas as pd

def streaming_kmeans(
    h5_path: str,
    table_key: str,
    n_clusters: int = 10,
    batch_size: int = 100_000,
    random_state: int = 42
) -> np.ndarray:
    """
    Incrementally cluster numeric rows from `table_key` in `h5_path` without
    holding the full DataFrame in memory.
    """
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state
    )
    # Partial‑fit on each chunk
    for chunk in pd.read_hdf(h5_path, table_key, iterator=True, chunksize=batch_size):
        numeric = chunk.select_dtypes(['int32', 'int64', 'float32', 'float64'])
        km.partial_fit(numeric.values)
    # Second pass to collect labels
    labels = []
    for chunk in pd.read_hdf(h5_path, table_key, iterator=True, chunksize=batch_size):
        numeric = chunk.select_dtypes(['int32', 'int64', 'float32', 'float64'])
        labels.append(km.predict(numeric.values))
    return np.concatenate(labels, axis=0)


