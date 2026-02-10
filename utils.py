import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred, task_type="Classification"):
    """
    Calculates metrics based on the task type.
    
    Args:
        y_true (array): True labels/values.
        y_pred (array): Predicted labels/values.
        task_type (str): 'Classification' or 'Regression'.
        
    Returns:
        dict: A dictionary of metric name and value.
    """
    metrics = {}
    if task_type == "Classification":
        acc = accuracy_score(y_true, y_pred)
        metrics["Accuracy"] = f"{acc:.4f}"
    else:
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics["Mean Squared Error (MSE)"] = f"{mse:.4f}"
        metrics["R2 Score"] = f"{r2:.4f}"
    
    return metrics

def get_algo_explanation(algo_name):
    """
    Returns a beginner-friendly explanation for the selected algorithm.
    """
    explanations = {
        "Linear Regression": "Finds the best-fitting straight line through the data points by minimizing the sum of squared errors.",
        "Logistic Regression": "Predicts the probability of a class (0 or 1) using a sigmoid function. It creates a linear decision boundary.",
        "Gradient Descent": "An optimization algorithm that iteratively adjusts parameters to minimize a cost function. Watch how the line moves towards the data!",
        "SVM": "Finds the 'hyperplane' that separates classes with the maximum margin (distance) to the nearest points (support vectors).",
        "KNN": "Classifies a point based on the 'vote' of its 'k' nearest neighbors. It adapts locally to the data structure.",
        "Decision Tree": "Splits data into branches based on feature values to create a flowchart-like structure for decision making.",
        "Random Forest": "An ensemble of many Decision Trees. It averages their predictions to reduce overfitting and improve accuracy.",
        "Naive Bayes": "Probabilistic classifier based on Bayes' Theorem. It assumes features are independent (hence 'Naive') and fits a distribution."
    }
    return explanations.get(algo_name, "Select an algorithm to learn more.")