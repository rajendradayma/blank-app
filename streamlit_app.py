import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import graphviz
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate Data
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Sidebar
st.sidebar.title("ML & DL Classifier")
classifier_name = st.sidebar.selectbox("Select Classifier", [
    "Decision Tree", "Logistic Regression", "Random Forest", "SVM",
    "KNN", "Naive Bayes", "MLP (Sklearn)", "Neural Network (Keras)"
])

# Parameters
params = {}
if classifier_name == "Decision Tree":
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 3)
    params["criterion"] = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
elif classifier_name == "Logistic Regression":
    params["C"] = st.sidebar.slider("Inverse Regularization (C)", 0.01, 10.0, 1.0)
elif classifier_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("Number of Estimators", 10, 100, 50)
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 3)
elif classifier_name == "SVM":
    params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
    params["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "rbf"])
elif classifier_name == "KNN":
    params["n_neighbors"] = st.sidebar.slider("Number of Neighbors", 1, 15, 5)
elif classifier_name == "MLP (Sklearn)":
    params["hidden_layer_sizes"] = st.sidebar.slider("Hidden Layer Size", 5, 100, 20)
elif classifier_name == "Neural Network (Keras)":
    params["hidden_units"] = st.sidebar.slider("Hidden Units", 4, 128, 16)
    params["epochs"] = st.sidebar.slider("Epochs", 10, 200, 50)
    params["batch_size"] = st.sidebar.slider("Batch Size", 8, 128, 32)

# Classifier Setup
def get_classifier(name, params):
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=params["max_depth"], criterion=params["criterion"])
    elif name == "Logistic Regression":
        return LogisticRegression(C=params["C"])
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"])
    elif name == "SVM":
        return SVC(C=params["C"], kernel=params["kernel"], probability=True)
    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    elif name == "Naive Bayes":
        return GaussianNB()
    elif name == "MLP (Sklearn)":
        return MLPClassifier(hidden_layer_sizes=(params["hidden_layer_sizes"],))
    else:
        return None

# Train & Predict
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.title("ML & DL Classification App")

if classifier_name != "Neural Network (Keras)":
    clf = get_classifier(classifier_name, params)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {acc:.2f}")
else:
    model = Sequential([
        Dense(params["hidden_units"], activation='relu', input_dim=2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=params["epochs"], batch_size=params["batch_size"], verbose=0)
    acc = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    y_proba = model.predict(X_test_scaled).ravel()
    y_pred = (y_proba > 0.5).astype(int)
    st.write(f"Accuracy: {acc:.2f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)

# Definition & Info
if classifier_name == "Decision Tree":
    with st.expander("Decision Tree - Definition & Pros/Cons"):
        st.markdown("""
        **Definition**: A Decision Tree is a tree-like structure used to make decisions based on features.
        
        **Formula**: Information Gain = Entropy(parent) - [Weighted average] * Entropy(children)

        **Pros**:
        - Easy to interpret
        - Handles both numerical and categorical data

        **Cons**:
        - Prone to overfitting
        - Unstable with small data changes
        """)

# Add similar blocks for other classifiers
