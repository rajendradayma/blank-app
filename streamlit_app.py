import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

st.set_page_config(layout="wide")
st.sidebar.title("ML Playground")

# Dataset
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Classifier Selection
classifier_name = st.sidebar.selectbox("Select Classifier", (
    "Decision Tree", "KNN", "Logistic Regression", "SVC", "Random Forest", "Naive Bayes"
))

# Classifier Parameters
def get_classifier(name):
    if name == "Decision Tree":
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
        return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    
    elif name == "KNN":
        n_neighbors = st.sidebar.slider("n_neighbors", 1, 15, 5)
        return KNeighborsClassifier(n_neighbors=n_neighbors)
    
    elif name == "Logistic Regression":
        C = st.sidebar.slider("Inverse Regularization (C)", 0.01, 10.0, 1.0)
        return LogisticRegression(C=C)
    
    elif name == "SVC":
        C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        probability = True
        return SVC(C=C, kernel=kernel, probability=probability)
    
    elif name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100)
        return RandomForestClassifier(n_estimators=n_estimators)
    
    elif name == "Naive Bayes":
        return GaussianNB()

# Draw Decision Boundary
def draw_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolor='k')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary")
    st.pyplot(fig)

# Plot ROC Curve
def plot_roc_curve(clf, X_test, y_test):
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        st.warning("ROC Curve is not available for this classifier.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Train and Evaluate
if st.sidebar.button("Run Classifier"):
    clf = get_classifier(classifier_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader(f"Classifier: {classifier_name}")
    st.write(f"Accuracy: {acc:.2f}")

    draw_decision_boundary(clf, X, y)
    plot_roc_curve(clf, X_test, y_test)

    if classifier_name == "Decision Tree":
        st.subheader("Decision Tree Structure")
        dot_data = export_graphviz(clf, out_file=None, feature_names=["Feature 1", "Feature 2"],
                                   class_names=["Class 0", "Class 1"], filled=True, rounded=True)
        st.graphviz_chart(dot_data)
