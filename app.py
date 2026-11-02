import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="DL-030: Neural Network vs Traditional ML Comparison", page_icon="🧬", layout="wide")

# Real results from the Keras notebook (Google Colab run), for the Results page.
NOTEBOOK = pd.DataFrame(
    [
        ["Logistic Regression", 0.9825, 0.9861, 0.9861, 0.9861, 0.0554, 0.9737, 0.0166],
        ["Random Forest", 0.9561, 0.9589, 0.9722, 0.9655, 1.5761, 0.9543, 0.0102],
        ["SVM (RBF)", 0.9825, 0.9861, 0.9861, 0.9861, 0.0167, 0.9772, 0.0163],
        ["Neural Network (Keras MLP)", 0.9649, 0.9857, 0.9583, 0.9718, 25.1465, 0.9667, 0.0170],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Train time (s)", "CV mean acc", "CV std"],
).set_index("Model")


@st.cache_resource(show_spinner="Training the four models (about a few seconds)...")
def get_models():
    return core.train_all()


models, times, (data, X_train, X_test, y_train, y_test) = get_models()

st.title("Neural Network vs Traditional ML Comparison")
st.caption("DL-030 | Breast Cancer Wisconsin | Priyansh Singh | Invertis University | Mentor: Mr. Raushan Raj")

page = st.sidebar.radio("Page", ["Predict", "Results", "About"])
st.sidebar.info("Educational demo only. Not a medical tool.")

if page == "Predict":
    st.subheader("Predict a tumour as malignant or benign")
    mode = st.radio("Input", ["Pick a sample from the test set", "Enter feature values"], horizontal=True)

    if mode == "Pick a sample from the test set":
        idx = st.selectbox("Test sample number", list(range(len(X_test))), index=0)
        x = X_test[idx]
        st.write(f"True label of this sample: **{core.CLASS_NAMES[int(y_test[idx])]}**")
        with st.expander("Show the 30 feature values"):
            st.dataframe(pd.DataFrame({"feature": data.feature_names, "value": x}), hide_index=True)
    else:
        st.write("Defaults are the training-set averages. Change any value and press the button.")
        means = X_train.mean(axis=0)
        cols = st.columns(3)
        vals = []
        for i, name in enumerate(data.feature_names):
            with cols[i % 3]:
                vals.append(st.number_input(name, value=float(means[i]), format="%.5f", key=f"f{i}"))
        x = vals

    if st.button("Predict", type="primary") or mode == "Pick a sample from the test set":
        out = core.predict_all(models, x)
        st.dataframe(
            out,
            hide_index=True,
            column_config={
                "P(benign)": st.column_config.ProgressColumn("P(benign)", min_value=0.0, max_value=1.0, format="%.3f"),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
            },
        )
        votes = out["Prediction"].value_counts()
        st.write("Models agree." if len(votes) == 1 else "Models disagree on this sample.")

elif page == "Results":
    st.subheader("Results from the Keras notebook (Colab run)")
    st.write("These are the numbers reported in the project: 114 test samples, 5-fold cross-validation.")
    st.dataframe(NOTEBOOK.style.format("{:.4f}"))
    st.bar_chart(NOTEBOOK[["Accuracy", "F1"]])
    st.markdown(
        "- Logistic Regression and SVM tied for best test accuracy (98.25%).\n"
        "- The neural network scored 96.49% and needed about 25 s to train.\n"
        "- The CV standard deviation (0.010 to 0.017) is larger than the gaps between the top models."
    )
    st.divider()
    st.subheader("Live check of the deployed models")
    st.caption(
        "The app retrains the models on the same split. Logistic Regression, Random Forest and SVM "
        "match the notebook. The neural network here is a scikit-learn MLP, so its numbers differ slightly."
    )
    live = core.evaluate(models, times, X_test, y_test).set_index("Model")
    st.dataframe(live.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1", "Train time (s)"]}))

else:
    st.subheader("About this project")
    st.markdown(
        """
**Question:** does a neural network give a real advantage over traditional machine learning on a small tabular dataset?

**Data:** Breast Cancer Wisconsin (Diagnostic): 569 samples, 30 numeric features, 2 classes (357 benign, 212 malignant).

**Setup:** one stratified 80/20 split (455 train, 114 test, seed 42) shared by all models. Scaling is fitted on the training set only.

**Models:** Logistic Regression, Random Forest, SVM (RBF) and a neural network (MLP 30-32-16-1).

**Finding:** on this small dataset the simple models matched or beat the neural network, which also trained far slower.

**Note on the deployed neural network:** the report uses a Keras MLP with dropout, trained in Google Colab
(see the notebook in the repository). To keep this app light, the deployed MLP uses scikit-learn with the same layer sizes,
Adam optimiser and early stopping, but no dropout layer.
"""
    )
