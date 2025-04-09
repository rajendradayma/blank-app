import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_curve, auc
import seaborn as sns

# Generate dataset
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Sidebar options
st.sidebar.title("Classifier Settings")
classifier_name = st.sidebar.selectbox("Select classifier", (
    "Decision Tree", "KNN", "Logistic Regression", "SVC", "Random Forest", "Naive Bayes"
))

# Hyperparameters
params = {}

if classifier_name == "Decision Tree":
    params['criterion'] = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
    params['splitter'] = st.sidebar.selectbox("Splitter", ["best", "random"])
    params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    params['min_samples_split'] = st.sidebar.slider("Min Samples Split", 2, 10, 2)
    params['min_samples_leaf'] = st.sidebar.slider("Min Samples Leaf", 1, 10, 1)

elif classifier_name == "KNN":
    params['n_neighbors'] = st.sidebar.slider("K", 1, 15)

elif classifier_name == "Logistic Regression":
    params['C'] = st.sidebar.number_input("Inverse Regularization (C)", 0.01, 10.0, 1.0)

elif classifier_name == "SVC":
    params['C'] = st.sidebar.number_input("Regularization (C)", 0.01, 10.0, 1.0)
    params['kernel'] = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])

elif classifier_name == "Random Forest":
    params['n_estimators'] = st.sidebar.slider("Number of Trees", 10, 100)
    params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20)

# Classifier selection
if classifier_name == "Decision Tree":
    model = DecisionTreeClassifier(**params)
elif classifier_name == "KNN":
    model = KNeighborsClassifier(**params)
elif classifier_name == "Logistic Regression":
    model = LogisticRegression(**params)
elif classifier_name == "SVC":
    model = SVC(probability=True, **params)
elif classifier_name == "Random Forest":
    model = RandomForestClassifier(**params)
elif classifier_name == "Naive Bayes":
    model = GaussianNB()

# Train
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Plot
st.subheader(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {accuracy:.2f}")

# Decision boundary plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3)
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
st.pyplot(fig)

# ROC Curve
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

# Definitions, Formulas, Pros & Cons
with st.expander(f"📘 About {classifier_name}"):
    if classifier_name == "Decision Tree":
        st.markdown("""
        **Definition**: Splits data using decision rules to classify.
        
        **Formula**:
        $$ Gini = 1 - \sum p_i^2 \quad \text{or} \quad Entropy = -\sum p_i \log_2(p_i) $$

        **Pros**: Easy to interpret, handles numerical/categorical data.

        **Cons**: Overfitting, unstable with small data changes.
        """)
    elif classifier_name == "KNN":
        st.markdown("""
        **Definition**: Predicts label based on majority of k-nearest neighbors.

        **Formula**:
        $$ d(x, y) = \sqrt{\sum (x_i - y_i)^2} $$

        **Pros**: Simple, no training. 

        **Cons**: Slow with large data, sensitive to irrelevant features.
        """)
    elif classifier_name == "Logistic Regression":
        st.markdown("""
        **Definition**: Models probability of class using logistic function.

        **Formula**:
        $$ P(y=1|x) = \frac{1}{1 + e^{-z}}, z = w^Tx + b $$

        **Pros**: Interpretable, efficient.

        **Cons**: Assumes linearity, poor with non-linear data.
        """)
    elif classifier_name == "SVC":
        st.markdown("""
        **Definition**: Finds optimal hyperplane for class separation.

        **Formula**:
        $$ \min \frac{1}{2}||w||^2 \text{ subject to } y_i(w^Tx_i + b) \geq 1 $$

        **Pros**: Good in high dimensions.

        **Cons**: Slow training, sensitive to parameters.
        """)
    elif classifier_name == "Random Forest":
        st.markdown("""
        **Definition**: Ensemble of decision trees, reducing variance.

        **Formula**:
        $$ \hat{y} = \text{majority\_vote}(Tree_1, ..., Tree_n) $$

        **Pros**: Reduces overfitting, handles missing data.

        **Cons**: Less interpretable, slow prediction.
        """)
    elif classifier_name == "Naive Bayes":
        st.markdown("""
        **Definition**: Uses Bayes Theorem assuming feature independence.

        **Formula**:
        $$ P(C|x) = \frac{P(C)\prod P(x_i|C)}{P(x)} $$

        **Pros**: Fast, works well on high dimensions.

        **Cons**: Strong independence assumption, poor with correlated features.
        """)
