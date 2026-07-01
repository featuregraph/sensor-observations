import os
import pyarrow.parquet as pq
import numpy as np

def load_observations(problem, dataset):
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "benchmark",
        problem,
        f"{dataset}.parquet"
    )
    path = os.path.abspath(path)

    table = pq.read_table(path)

    return np.stack([
        table["state"].to_numpy(),
        table["measurement"].to_numpy(),
    ], axis=1)
