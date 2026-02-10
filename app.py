import streamlit as st
import numpy as np

# Import custom modules
import data
import algorithms
import visualizations
import utils

# --- Page Config ---
st.set_page_config(
    page_title="ML Algorithm Visualizer",
    page_icon="📊",
    layout="wide"
)

# --- Sidebar Controls ---
st.sidebar.header("1. Dataset Configuration")

# Dataset Selection
task_type = st.sidebar.radio("Select Task Type", ["Classification", "Regression"])

if task_type == "Regression":
    dataset_name = "Regression (Linear)"
    algo_options = ["Linear Regression", "Gradient Descent"]
else:
    dataset_name = st.sidebar.selectbox("Select Dataset Shape", ["Classification (Blobs)", "Classification (Moons)"])
    algo_options = ["Logistic Regression", "SVM", "KNN", "Decision Tree", "Random Forest", "Naive Bayes"]

# Data Parameters
n_samples = st.sidebar.slider("Number of Samples", 50, 500, 200, step=50)
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.2, step=0.05)

# Generate Data
X, y = data.get_dataset(dataset_name, n_samples, noise_level)

st.sidebar.markdown("---")
st.sidebar.header("2. Algorithm Selection")
selected_algo = st.sidebar.selectbox("Select Algorithm", algo_options)

# --- Algorithm Hyperparameters ---
params = {}
if selected_algo == "KNN":
    params["n_neighbors"] = st.sidebar.slider("K (Neighbors)", 1, 15, 5)
elif selected_algo == "SVM":
    params["C"] = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
    params["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
elif selected_algo in ["Decision Tree", "Random Forest"]:
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 15, 5)
    if selected_algo == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Num Trees", 10, 100, 50)
elif selected_algo == "Gradient Descent":
    params["learning_rate"] = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
    params["epochs"] = st.sidebar.slider("Epochs", 10, 500, 100)
elif selected_algo == "Logistic Regression":
    params["C"] = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)

# --- Main Layout ---
st.title("🤖 ML Algorithm Visualizer")
st.markdown("""
Welcome! This tool helps you **visualize** how different Machine Learning algorithms learn patterns in data.
1. Use the **Sidebar** to generate data and select an algorithm.
2. Tweak **Hyperparameters** to see how the decision boundary or regression line changes.
""")

# --- Model Training & Prediction ---
model = algorithms.train_model(selected_algo, X, y, params)
y_pred = algorithms.predict_model(model, X)

# --- Visualization & Results ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Visualization: {selected_algo}")
    
    if task_type == "Regression":
        fig = visualizations.plot_regression(X, y, model, selected_algo)
    else:
        fig = visualizations.plot_decision_boundary(X, y, model, selected_algo)
        
    st.pyplot(fig)

with col2:
    st.subheader("Performance & Info")
    
    # Display Metrics
    metrics = utils.calculate_metrics(y, y_pred, task_type)
    for metric, value in metrics.items():
        st.metric(label=metric, value=value)
    
    st.markdown("---")
    st.markdown("**Algorithm Logic:**")
    st.info(utils.get_algo_explanation(selected_algo))
    
    st.markdown("**Parameter Impact:**")
    if selected_algo == "KNN":
        st.caption("Low 'K' captures local noise (overfitting). High 'K' smooths the boundary (underfitting).")
    elif selected_algo == "SVM":
        st.caption("High 'C' tries to classify every point correctly (complex boundary). Low 'C' allows some errors (smoother boundary).")
    elif selected_algo == "Decision Tree":
        st.caption("Deeper trees capture more complex patterns but risk overfitting.")
    elif selected_algo == "Gradient Descent":
        st.caption("Learning rate controls step size. Too high = overshoot. Too low = slow convergence.")
    else:
        st.caption("Adjust parameters in the sidebar to see their effect.")

# --- Educational Footer ---
st.markdown("---")
st.markdown("### 🧠 Did you know?")
st.write(f"You are currently using **{selected_algo}**. Try changing the 'Noise Level' in the dataset controls to see how robust this algorithm is against messy data!")