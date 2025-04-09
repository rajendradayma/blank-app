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
st.sidebar.title("ML Playground with Parameters")

# Load dataset
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

classifier_name = st.sidebar.selectbox("Select Classifier", (
    "Decision Tree", "KNN", "Logistic Regression", "SVC", "Random Forest", "Naive Bayes"
))

# Parameter setup
def get_classifier(name):
    if name == "Decision Tree":
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        splitter = st.sidebar.selectbox("Splitter", ["best", "random"])
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 3)
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
        min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
        max_features = st.sidebar.selectbox("Max Features", [None, "sqrt", "log2"])
        max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes", 2, 100, 10)
        min_impurity_decrease = st.sidebar.number_input("Min Impurity Decrease", 0.0, 1.0, 0.0)
        return DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            random_state=42
        )

    elif name == "KNN":
        n_neighbors = st.sidebar.slider("n_neighbors", 1, 30, 5)
        weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
        algorithm = st.sidebar.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

    elif name == "Logistic Regression":
        C = st.sidebar.slider("Inverse Regularization (C)", 0.01, 10.0, 1.0)
        solver = st.sidebar.selectbox("Solver", ["liblinear", "lbfgs", "newton-cg", "sag", "saga"])
        max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 100)
        return LogisticRegression(C=C, solver=solver, max_iter=max_iter)

    elif name == "SVC":
        C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
        return SVC(C=C, kernel=kernel, gamma=gamma, probability=True)

    elif name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100)
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        max_features = st.sidebar.selectbox("Max Features", ["auto", "sqrt", "log2"])
        bootstrap = st.sidebar.checkbox("Bootstrap", True)
        return RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                      max_depth=max_depth, max_features=max_features,
                                      bootstrap=bootstrap, random_state=42)

    elif name == "Naive Bayes":
        return GaussianNB()

# Plotting functions
def draw_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolor='k')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary")
    st.pyplot(fig)

def plot_roc_curve(clf, X_test, y_test):
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        st.warning("ROC Curve not available for this model.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
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
    st.write(f"Accuracy: **{acc:.2f}**")

    draw_decision_boundary(clf, X, y)
    plot_roc_curve(clf, X_test, y_test)

    if classifier_name == "Decision Tree":
        st.subheader("Decision Tree Visualization")
        dot_data = export_graphviz(clf, out_file=None, feature_names=["X1", "X2"],
                                   class_names=["0", "1"], filled=True, rounded=True)
        st.graphviz_chart(dot_data)

