
def geodesic_mse(X, Y, eps=1e-7):
    """
    Computes MSE using geodesic (arc) distance on the sphere.
    Assumes X and Y are [n, dim] and lie on the unit sphere.
    """
    assert X.shape == Y.shape
    dot_products = np.sum(X * Y, axis=1)
    dot_products = np.clip(dot_products, -1.0 + eps, 1.0 - eps)
    angles = np.arccos(dot_products)
    mse = np.mean(angles ** 2)
    return mse