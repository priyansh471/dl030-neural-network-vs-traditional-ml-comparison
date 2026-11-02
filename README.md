# DL-030: Neural Network vs Traditional ML Comparison

Does a neural network give a real advantage over traditional machine learning on a small tabular dataset?
This project compares Logistic Regression, Random Forest, SVM and a neural network (MLP) on the
Breast Cancer Wisconsin (Diagnostic) dataset.

**Student:** Priyansh Singh | B.Tech CSE, Invertis University | Mentor: Mr. Raushan Raj | IIOT-5 Deep Learning

## Results (Keras notebook, Google Colab)

114 test samples, one stratified 80/20 split (seed 42).

| Model | Accuracy | Precision | Recall | F1 | Train time (s) | CV mean acc |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.06 | 0.9737 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 1.58 | 0.9543 |
| SVM (RBF) | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.02 | 0.9772 |
| Neural Network (Keras MLP) | 0.9649 | 0.9857 | 0.9583 | 0.9718 | 25.15 | 0.9667 |

Precision and recall are for the benign class (label 1). Missed malignant cases: LR 1, RF 3, SVM 1, NN 1.

**Finding:** on this small dataset the simple models matched or beat the neural network, which also trained far slower.
The cross-validation spread (0.010 to 0.017) is larger than the gaps between the top models.

## Repository contents

| File | What it is |
|---|---|
| `DL030_NN_vs_Traditional_ML.ipynb` | The full experiment (Keras MLP, plots, cross-validation). Run it in Google Colab. |
| `app.py` | Streamlit demo app: Predict, Results and About pages. |
| `core.py` | Data split, models, evaluation and prediction code used by the app. |
| `requirements.txt` | Python packages for the app. |

## Run the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick this repository, branch `main`, and main file `app.py`.
4. Click **Deploy**. The link to share is shown when it finishes.

## Note on the deployed neural network

The report uses a Keras MLP with dropout (see the notebook). To keep the app small and fast to deploy,
the app uses a scikit-learn `MLPClassifier` with the same layer sizes (30-32-16-1), ReLU, Adam (learning rate 0.001),
batch size 16 and early stopping, but no dropout layer. Its numbers differ slightly from the Keras results.

## Disclaimer

Educational project. Not a medical device and not for clinical use.
