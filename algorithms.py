import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def train_model(algo_name, X, y, params):
    """
    Trains the selected model with given parameters.
    
    Args:
        algo_name (str): Name of the algorithm.
        X (array): Feature data.
        y (array): Target labels.
        params (dict): Hyperparameters from UI.
        
    Returns:
        model: Trained model object (sklearn or custom).
    """
    
    model = None

    if algo_name == "Linear Regression":
        # Reshape X for sklearn (needs 2D array)
        X_reshaped = X.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_reshaped, y)
        
    elif algo_name == "Gradient Descent":
        # Custom implementation for educational purposes
        # Returns a dictionary containing history for animation/plotting
        lr = params.get("learning_rate", 0.01)
        epochs = params.get("epochs", 100)
        
        m = 0 # Slope
        b = 0 # Intercept
        n = float(len(X))
        history = []
        
        for i in range(epochs):
            y_pred = m * X + b
            # Derivatives
            d_m = (-2/n) * sum(X * (y - y_pred))
            d_b = (-2/n) * sum(y - y_pred)
            # Update
            m = m - lr * d_m
            b = b - lr * d_b
            
            # Save history every few epochs to avoid clutter
            if i % 10 == 0 or i == epochs - 1:
                history.append((m, b))
                
        model = {"type": "CustomGD", "m": m, "b": b, "history": history}

    elif algo_name == "Logistic Regression":
        C_val = params.get("C", 1.0)
        model = LogisticRegression(C=C_val)
        model.fit(X, y)

    elif algo_name == "SVM":
        C_val = params.get("C", 1.0)
        kernel = params.get("kernel", "rbf")
        model = SVC(C=C_val, kernel=kernel)
        model.fit(X, y)

    elif algo_name == "KNN":
        k = params.get("n_neighbors", 5)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)

    elif algo_name == "Decision Tree":
        depth = params.get("max_depth", 5)
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X, y)

    elif algo_name == "Random Forest":
        n_est = params.get("n_estimators", 100)
        depth = params.get("max_depth", 5)
        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth)
        model.fit(X, y)
        
    elif algo_name == "Naive Bayes":
        model = GaussianNB()
        model.fit(X, y)

    return model

def predict_model(model, X):
    """
    Uniform prediction interface.
    """
    if isinstance(model, dict) and model.get("type") == "CustomGD":
        # Custom Gradient Descent prediction
        m = model["m"]
        b = model["b"]
        return m * X + b
    
    # Sklearn models
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return model.predict(X)