# 📊 ML Algorithm Visualizer
**Developed by: Bushra-aqib**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An interactive, educational machine learning tool designed to provide a visual bridge between complex mathematical models and intuitive understanding. This application allows users to manipulate data and parameters in real-time to see how algorithms "think."

---

## 📂 Project Structure

The project follows a modular architecture to ensure clean code and easy maintainability:

```text
ml_visualizer/
├── app.py               # Main UI and application controller
├── data.py              # Synthetic dataset generation logic
├── algorithms.py        # ML implementations (Scratch & Scikit-learn)
├── visualizations.py    # Matplotlib & Seaborn plotting functions
├── utils.py             # Performance metrics and helper functions
└── README.md            # Project documentation
```
---
## 📈 Information About Datasets
The application utilizes synthetic datasets generated dynamically via scikit-learn. This allows users to test algorithm limits in controlled environments:

- Classification (Blobs): Linearly separable clusters used to demonstrate basic class boundaries.

- Classification (Moons): Non-linear, interlocking crescent shapes designed to test the robustness of complex models like RBF-SVM or Random Forests.

- Linear Regression Data: Single-feature continuous data used to demonstrate line fitting and error minimization.

## ✨ Key Features

- Real-Time Parameter Tuning: Use sliders to adjust learning rates, $K$ values, tree depth, and regularization ($C$) on the fly.
  
- Dynamic Decision Boundaries: Visualize how models carve the feature space to separate classes.
  
- Loss Visualization: Watch Gradient Descent iterate through epochs as the regression line converges on the data.

- Performance Tracking: Live updates of Accuracy, MSE, and $R^2$ Score as parameters change.
  ---

## 🧠 Algorithms Implemented

**1. Classification**

- Logistic Regression:Linear boundary using the Sigmoid function.

- Support Vector Machine (SVM): Maximum margin separation with Linear, RBF, and Polynomial kernels.

- K-Nearest Neighbors (KNN): A distance-based approach demonstrating the "Goldilocks" zone of $K$.

- Decision Tree & Random Forest: Hierarchical splitting and ensemble learning.

- Naive Bayes: Probability-based classification using Gaussian distributions.

**2. Regression**

- Linear Regression: Ordinary Least Squares fitting

- Gradient Descent (From Scratch): A manual implementation of the optimization algorithm showing how $m$ (slope) and $b$ (intercept) are updated iteratively.

---

## 💻 Important Commands
**1. Installation**
Ensure you have Python installed, then install the necessary dependencies:

```Bash
pip install streamlit numpy matplotlib scikit-learn seaborn
```
**2. Running the App**
Navigate to the project root and execute the following command:

```Bash
streamlit run app.py
```
