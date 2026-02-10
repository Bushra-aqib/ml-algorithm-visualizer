import numpy as np
from sklearn.datasets import make_classification, make_regression, make_moons

def get_dataset(dataset_name, n_samples, noise):
    """
    Generates synthetic datasets for visualization.
    
    Args:
        dataset_name (str): The type of dataset to generate.
        n_samples (int): Number of data points.
        noise (float): Amount of noise/overlap in data.
        
    Returns:
        X (array): Features.
        y (array): Target.
    """
    # Fix random state for reproducibility during visualization
    rng_state = 42
    
    if dataset_name == "Regression (Linear)":
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise*10, random_state=rng_state)
        # Flatten for consistency
        X = X.flatten() 
        
    elif dataset_name == "Classification (Blobs)":
        # Standard two-cluster problem
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0, 
            n_informative=2, n_clusters_per_class=1, 
            flip_y=noise/5, random_state=rng_state
        )
        
    elif dataset_name == "Classification (Moons)":
        # Non-linear dataset (two swirling moons)
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=rng_state)
        
    else:
        return None, None

    return X, y