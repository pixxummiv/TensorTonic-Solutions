import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x)
    p = np.asarray(p)
    if np.allclose(np.sum(p), 1, rtol=1e-6):
        return np.sum(x*p)
    else:
        raise ValueError
