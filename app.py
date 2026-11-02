import altair as alt
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="DL-030: Neural Network vs Traditional ML Comparison", page_icon="🧬", layout="wide")

# Real results from the Keras notebook (Google Colab run).
NOTEBOOK = pd.DataFrame(
    [
        ["Logistic Regression", 0.9825, 0.9861, 0.9861, 0.9861, 0.0554, 0.9737, 0.0166],
        ["Random Forest", 0.9561, 0.9589, 0.9722, 0.9655, 1.5761, 0.9543, 0.0102],
        ["SVM (RBF)", 0.9825, 0.9861, 0.9861, 0.9861, 0.0167, 0.9772, 0.0163],
        ["Neural Network (Keras MLP)", 0.9649, 0.9857, 0.9583, 0.9718, 25.1465, 0.9667, 0.0170],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Train time (s)", "CV mean acc", "CV std"],
).set_index("Model")


@st.cache_resource(show_spinner="Training the four models (a few seconds)...")
def get_models():
    return core.train_all()


models, times, (data, X_train, X_test, y_train, y_test) = get_models()

# ---------------------------------------------------------------- Header
st.title("Neural Network vs Traditional ML Comparison")
st.caption("DL-030 | Priyansh Singh | Invertis University | Mentor: Mr. Raushan Raj | Educational demo, not a medical tool")

# ---------------------------------------------------------------- About
st.markdown(
    """
**Question:** does a neural network give a real advantage over traditional machine learning on a small tabular dataset?

**Data:** Breast Cancer Wisconsin (Diagnostic), 569 samples, 30 features, 357 benign and 212 malignant.

**Setup:** one stratified 80/20 split (455 train, 114 test, seed 42) shared by all four models.

**Finding:** the simple models matched or beat the neural network, which also trained far slower.
"""
)

st.divider()

# ---------------------------------------------------------------- Predict
st.header("1. Try it: predict malignant or benign")
mode = st.radio("Input", ["Pick a sample from the test set", "Enter feature values"], horizontal=True)

if mode == "Pick a sample from the test set":
    idx = st.selectbox("Test sample number", list(range(len(X_test))), index=0)
    x = X_test[idx]
    st.write(f"True label of this sample: **{core.CLASS_NAMES[int(y_test[idx])]}**")
    with st.expander("Show the 30 feature values"):
        st.dataframe(pd.DataFrame({"feature": data.feature_names, "value": x}), hide_index=True)
    run = True
else:
    st.write("Defaults are the training-set averages. Change any value and press Predict.")
    means = X_train.mean(axis=0)
    cols = st.columns(3)
    vals = []
    for i, name in enumerate(data.feature_names):
        with cols[i % 3]:
            vals.append(st.number_input(name, value=float(means[i]), format="%.5f", key=f"f{i}"))
    x = vals
    run = st.button("Predict", type="primary")

if run:
    out = core.predict_all(models, x)
    st.dataframe(
        out,
        hide_index=True,
        column_config={
            "P(benign)": st.column_config.ProgressColumn("P(benign)", min_value=0.0, max_value=1.0, format="%.3f"),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
        },
    )
    st.write("All four models agree." if out["Prediction"].nunique() == 1 else "The models disagree on this sample.")

st.divider()

# ---------------------------------------------------------------- Results
st.header("2. Results from the Keras notebook (Colab run)")
st.caption("114 test samples and 5-fold cross-validation. Precision and recall are for the benign class.")
st.dataframe(NOTEBOOK.style.format("{:.4f}"))

chart_df = NOTEBOOK.reset_index()[["Model", "Accuracy"]]
bars = alt.Chart(chart_df, title="Test accuracy (axis starts at 0.90)").mark_bar(clip=True).encode(
    x=alt.X("Model:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
    y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0.9, 1.0], zero=False)),
    color=alt.Color("Model:N", legend=None),
    tooltip=["Model", alt.Tooltip("Accuracy:Q", format=".4f")],
)
labels = bars.mark_text(dy=-8, clip=False).encode(text=alt.Text("Accuracy:Q", format=".4f"))
st.altair_chart(bars + labels, use_container_width=True)

st.markdown(
    "- Logistic Regression and SVM tied for best test accuracy (98.25%).\n"
    "- The neural network scored 96.49% and needed about 25 s to train, versus 0.06 s for Logistic Regression.\n"
    "- The CV standard deviation (0.010 to 0.017) is larger than the gaps between the top models."
)

with st.expander("Live check of the models deployed in this app"):
    st.caption(
        "The app retrains the models on the same split. Logistic Regression, Random Forest and SVM match the notebook. "
        "The neural network here is a scikit-learn MLP (same 30-32-16-1 layers, Adam, early stopping, no dropout), "
        "so its numbers differ slightly from the Keras result."
    )
    live = core.evaluate(models, times, X_test, y_test).set_index("Model")
    st.dataframe(live.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1", "Train time (s)"]}))
