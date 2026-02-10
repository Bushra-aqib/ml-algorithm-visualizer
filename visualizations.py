import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_regression(X, y, model, algo_name):
    """
    Plots Linear Regression and Gradient Descent results.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot raw data
    ax.scatter(X, y, color='blue', alpha=0.6, label='Data Points')
    
    # Plot predictions
    X_range = np.linspace(min(X), max(X), 100)
    
    if algo_name == "Gradient Descent":
        # Plot the final line
        m = model["m"]
        b = model["b"]
        y_line = m * X_range + b
        ax.plot(X_range, y_line, color='red', linewidth=3, label='Final Model')
        
        # Optional: Visualize previous iterations faintly
        history = model["history"]
        for i, (m_hist, b_hist) in enumerate(history[:-1]):
            # Only plot a few to keep it clean
            if i % (len(history)//5 + 1) == 0:
                y_hist = m_hist * X_range + b_hist
                ax.plot(X_range, y_hist, color='green', alpha=0.2, linestyle='--')
                
        ax.set_title(f"Gradient Descent: y = {m:.2f}x + {b:.2f}")
        
    else:
        # Sklearn Linear Regression
        y_line = model.predict(X_range.reshape(-1, 1))
        ax.plot(X_range, y_line, color='red', linewidth=3, label='Prediction Line')
        ax.set_title("Linear Regression Fit")

    ax.set_xlabel("Feature X")
    ax.set_ylabel("Target y")
    ax.legend()
    return fig

def plot_decision_boundary(X, y, model, algo_name):
    """
    Plots decision boundaries for classification algorithms.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Setup grid for contour plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    # Predict on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contour (Decision Boundary)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors='k', cmap='coolwarm', alpha=0.8)
    
    # Legend and labels
    ax.set_title(f"Decision Boundary: {algo_name}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    
    return fig
