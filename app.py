import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(layout="wide")
st.title("🎓 ML Pipeline Dashboard - Student Performance")

st.markdown("""
<style>
.divider {
    border-right: 3px solid #444;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(r"C:\ANN_ML\student_performance_dataset_v2.csv")

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data & EDA",
    "🧹 Data Engineering",
    "🎯 Feature Engineering",
    "🤖 ML Pipeline",
    "📈 Performance"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:

    st.subheader("📊 Exploratory Data Analysis")

    # Layout with divider
    left, divider, right = st.columns([1, 0.05, 2])

    # -------- LEFT SIDE --------
    with left:
        st.markdown("### 📄 Dataset Summary")

        st.write("**Shape:**", df.shape)

        st.markdown("**Columns:**")
        st.write(list(df.columns))

        st.markdown("**Preview:**")
        st.dataframe(df.head(), height=200)

        target = st.selectbox("🎯 Select Target Variable", df.columns)
        st.session_state["target"] = target

    # -------- DIVIDER --------
    with divider:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # -------- RIGHT SIDE --------
    with right:
        st.markdown("### 📈 Visual Analysis")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.hist(df[target])
            ax.set_title("Distribution")
            st.pyplot(fig, use_container_width=True)

        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[target], ax=ax)
            ax.set_title("Boxplot")
            st.pyplot(fig, use_container_width=True)

        st.markdown("### 🔥 Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(df.corr(numeric_only=True), ax=ax)
        st.pyplot(fig, use_container_width=True)
        
# =========================================================
# TAB 2
# =========================================================
with tab2:
    numeric_cols = df.select_dtypes(include=np.number).columns

    st.write("Zero Count:")
    st.write((df[numeric_cols] == 0).sum())

    action = st.selectbox("Handle Zeros", ["Keep", "Delete Rows", "Median"])

    df_clean = df.copy()

    if action == "Delete Rows":
        df_clean = df_clean[(df_clean[numeric_cols] != 0).all(axis=1)]

    elif action == "Median":
        for col in numeric_cols:
            df_clean[col] = df_clean[col].replace(0, df_clean[col].median())

    if st.checkbox("Remove Outliers"):
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            df_clean = df_clean[
                (df_clean[col] >= Q1 - 1.5*IQR) &
                (df_clean[col] <= Q3 + 1.5*IQR)
            ]

    st.write("Current Shape:", df_clean.shape)

    st.session_state["df_clean"] = df_clean

# =========================================================
# TAB 3
# =========================================================
with tab3:

    if "df_clean" not in st.session_state or "target" not in st.session_state:
        st.warning("Complete previous steps first")
    else:
        df_clean = st.session_state["df_clean"]
        target = st.session_state["target"]

        X = df_clean.drop(columns=[target])
        y_raw = df_clean[target]

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        X_encoded = pd.get_dummies(X, drop_first=True)

        method = st.selectbox("Feature Selection", ["All", "Variance", "Information Gain"])

        if method == "Variance":
            selector = VarianceThreshold(0.1)
            X_sel = selector.fit_transform(X_encoded)
            cols = X_encoded.columns[selector.get_support()]
            X_selected = pd.DataFrame(X_sel, columns=cols)

        elif method == "Information Gain":
            scores = mutual_info_classif(X_encoded, y)
            cols = X_encoded.columns[scores > np.mean(scores)]
            X_selected = X_encoded[cols]

        else:
            X_selected = X_encoded
            cols = X_encoded.columns

        st.write("Selected Features:", list(cols))
        st.write(X_selected.head())

        st.session_state["X_selected"] = X_selected
        st.session_state["y"] = y

# =========================================================
# TAB 4
# =========================================================
with tab4:

    st.subheader("Training Configuration")

    if "X_selected" not in st.session_state:
        st.warning("⚠️ Go to Feature Engineering tab first")

    model_name = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )

    test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

    if st.button("🚀 Start Training Pipeline"):

        if "X_selected" not in st.session_state:
            st.error("❌ Complete Feature Engineering first")

        else:
            st.write("Training Started...")

            X = st.session_state["X_selected"]
            y = st.session_state["y"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            else:
                model = RandomForestClassifier()

            st.write("Model:", model_name)

            scores = cross_val_score(model, X, y, cv=5)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state["results"] = {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1": f1_score(y_test, y_pred, average='weighted'),
                "cv": scores
            }

            st.success("✅ Training Completed!")

# =========================================================
# TAB 5
# =========================================================
with tab5:

    if "results" in st.session_state:

        res = st.session_state["results"]

        st.subheader("Model Performance")

        st.write("Model:", res["model"])
        st.write("Accuracy:", round(res["accuracy"], 3))
        st.write("Precision:", round(res["precision"], 3))
        st.write("Recall:", round(res["recall"], 3))
        st.write("F1 Score:", round(res["f1"], 3))

        st.write("Average CV Score:", round(np.mean(res["cv"]), 3))

        st.line_chart(res["cv"])

    else:
        st.warning("Run training first")