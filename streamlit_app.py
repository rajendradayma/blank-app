import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Function to create mesh grid
def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

# Streamlit UI
st.title("🧠 ML Classifier Playground")
st.sidebar.title("Model Settings")

algorithm = st.sidebar.selectbox(
    'Select Classifier',
    ('Decision Tree', 'Logistic Regression', 'K-Nearest Neighbors', 'Random Forest', 'SVM', 'Naive Bayes')
)

params = {}

if algorithm == 'Decision Tree':
    params["criterion"] = st.sidebar.selectbox("Criterion", ['gini', 'entropy'])
    params["splitter"] = st.sidebar.selectbox("Splitter", ['best', 'random'])
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, value=5)
    params["min_samples_split"] = st.sidebar.slider("Min Samples Split", 2, 10, value=2)
    params["min_samples_leaf"] = st.sidebar.slider("Min Samples Leaf", 1, 10, value=1)
elif algorithm == 'Logistic Regression':
    params["C"] = st.sidebar.number_input("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0)
    params["max_iter"] = st.sidebar.slider("Max Iterations", 100, 1000, 200)
elif algorithm == 'K-Nearest Neighbors':
    params["n_neighbors"] = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 5)
    params["weights"] = st.sidebar.selectbox("Weights", ['uniform', 'distance'])
elif algorithm == 'Random Forest':
    params["n_estimators"] = st.sidebar.slider("Number of Estimators", 10, 200, 100)
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, value=5)
elif algorithm == 'SVM':
    params["C"] = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    params["kernel"] = st.sidebar.selectbox("Kernel", ['linear', 'rbf', 'poly'])
elif algorithm == 'Naive Bayes':
    st.sidebar.info("No hyperparameters to tune for GaussianNB.")

# Classifier setup
def get_classifier(name, params):
    if name == 'Decision Tree':
        return DecisionTreeClassifier(**params, random_state=42)
    elif name == 'Logistic Regression':
        return LogisticRegression(**params)
    elif name == 'K-Nearest Neighbors':
        return KNeighborsClassifier(**params)
    elif name == 'Random Forest':
        return RandomForestClassifier(**params, random_state=42)
    elif name == 'SVM':
        return SVC(**params, probability=True)
    elif name == 'Naive Bayes':
        return GaussianNB()

clf = get_classifier(algorithm, params)

# Plot base data
fig, ax = plt.subplots()
ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
plt.xlabel("Col1")
plt.ylabel("Col2")
orig = st.pyplot(fig)

if st.sidebar.button("Run Algorithm"):
    orig.empty()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    ax.set_xlabel("Col1")
    ax.set_ylabel("Col2")
    orig = st.pyplot(fig)

    st.subheader(f"Accuracy: **{accuracy_score(y_test, y_pred):.2f}**")

    # Tree visualization
    if algorithm == "Decision Tree":
        tree_graph = export_graphviz(clf, feature_names=["Col1", "Col2"])
        st.graphviz_chart(tree_graph)
