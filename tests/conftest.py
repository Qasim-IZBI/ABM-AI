import os
import tempfile
import numpy as np
import pytest


@pytest.fixture(scope="session")
def tmp_data_dir():
    """Create a minimal processed data directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "inputs.npy"),
                np.random.rand(20, 2).astype(np.float32))
        np.save(os.path.join(d, "labels.npy"),
                np.random.rand(20, 6).astype(np.float32))
        with open(os.path.join(d, "image_paths.txt"), "w") as f:
            f.write("\n".join(["MISSING"] * 20))
        yield d
