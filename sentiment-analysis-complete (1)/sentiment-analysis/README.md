# AI-Powered Sentiment Analysis System

A supervised Natural Language Processing (NLP) and machine-learning project that classifies customer reviews into **Positive**, **Negative**, or **Neutral** sentiment. The project covers the complete workflow—from data inspection and text preprocessing to model evaluation, serialization, and an interactive **Streamlit** application.

> **Project level:** Beginner to Intermediate  
> **Project type:** NLP + Machine Learning  
> **Dataset:** 10,000 customer-review records  
> **Final model:** TF-IDF + tuned Logistic Regression

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Evaluation](#model-evaluation)
- [Run the Project Locally](#run-the-project-locally)
- [Use the Streamlit Application](#use-the-streamlit-application)
- [Retrain the Model](#retrain-the-model)
- [Project Outputs](#project-outputs)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Learning Outcomes](#learning-outcomes)

## Project Overview

Customer reviews contain useful feedback about products and services, but manually reading and categorizing large numbers of reviews is time-consuming. This project builds a machine-learning system that automatically predicts the sentiment expressed in a review.

The application accepts a review as text and returns one of three classes:

| Sentiment | Meaning |
|---|---|
| **Positive** | The review expresses satisfaction or a favourable opinion. |
| **Negative** | The review expresses dissatisfaction or an unfavourable opinion. |
| **Neutral** | The review is mainly factual, informational, or does not express a clear positive or negative opinion. |

The emphasis is on a transparent, conventional machine-learning workflow. The project does not require a large language model, generative AI service, or external inference API.

## Features

- Dataset validation, cleaning, and exploratory data analysis (EDA).
- Review-length, sentiment, category, and source visualizations.
- Conservative text normalization that retains sentiment-changing negation words.
- Bag-of-Words (BoW) and TF-IDF feature extraction.
- Comparison of Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machine (SVM) models.
- Group-aware train/test splitting and cross-validation to reduce identical-text leakage.
- Logistic Regression hyperparameter tuning with `GridSearchCV`.
- Evaluation using accuracy, macro precision, macro recall, macro F1-score, confusion matrix, and multiclass ROC-AUC.
- A serialized scikit-learn pipeline for inference without retraining at application startup.
- A Streamlit interface for entering a review and viewing its predicted sentiment and estimated class probabilities.

## Technology Stack

| Tool / Library | Purpose |
|---|---|
| Python | Core programming language |
| Pandas, NumPy | Data loading, cleaning, and numerical operations |
| Scikit-learn | Text vectorization, model training, validation, tuning, and evaluation |
| Matplotlib, Seaborn | EDA and evaluation visualizations |
| Joblib | Saving and loading the trained pipeline |
| Streamlit | Interactive web application |
| Jupyter Notebook | Exploratory analysis and documentation of the workflow |

## Repository Structure

```text
sentiment-analysis/
├── data/
│   └── reviews.csv                     # 10,000-row input dataset
├── models/
│   └── sentiment_pipeline.joblib       # Serialized TF-IDF + classifier pipeline
├── notebooks/
│   └── sentiment_analysis.ipynb        # End-to-end analysis notebook
├── outputs/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── roc_curves.png
├── reports/
│   ├── evaluation_report.pdf
│   └── evaluation_report.docx
├── app.py                              # Streamlit application
├── sentiment_utils.py                  # Shared text-normalization function
├── train_model.py                      # Reproducible training and evaluation script
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

Additional CSV, JSON, and chart outputs are generated in `outputs/` when `train_model.py` runs. The training script also writes a model-comparison CSV into `reports/`.

## Dataset

The supplied dataset is included at `data/reviews.csv` and contains **10,000 records** with the following columns:

| Column | Description |
|---|---|
| `review_id` | Unique identifier for a review record |
| `review_text` | Customer review text; primary model input |
| `sentiment` | Target class: `Positive`, `Negative`, or `Neutral` |
| `category` | Product or service category |
| `source` | Source of the feedback |

### Dataset observations

- The three sentiment classes are almost perfectly balanced: Positive = 3,334, Negative = 3,333, and Neutral = 3,333.
- The supplied file has no missing values or exact duplicate rows.
- **Important:** the 10,000 records contain only **300 distinct review-text values**. Many records repeat the same review text while differing in identifiers or metadata.

The repeated text is a significant limitation of the dataset. To avoid placing an identical review text in both the training and test partitions, `review_text` is used as the grouping variable during splitting and cross-validation. This reduces direct text leakage, but it does not make the dataset representative of naturally varied customer language.

## Methodology

The project follows these stages:

1. **Data understanding and cleaning** – Load the CSV, validate required columns and labels, inspect missing values and duplicates, and remove blank review texts.
2. **Exploratory data analysis** – Examine sentiment-class counts, review lengths, product/service categories, and feedback sources.
3. **Text preprocessing** – Decode HTML entities, remove HTML tags, replace URLs with a token, convert text to lowercase, remove most punctuation, and normalize whitespace. Negation words such as `not` and `no` are retained because they can change the meaning of a review.
4. **Feature extraction** – Convert text into numerical features using TF-IDF and Bag of Words. Vectorizers are kept inside scikit-learn pipelines so that vocabulary fitting occurs only on the relevant training folds.
5. **Model comparison** – Evaluate these baseline configurations:
   - Multinomial Naive Bayes with TF-IDF
   - Logistic Regression with TF-IDF
   - Linear SVM with TF-IDF
   - Logistic Regression with Bag of Words
6. **Validation and tuning** – Use `StratifiedGroupKFold` with `review_text` as the group. Tune the TF-IDF + Logistic Regression pipeline using `GridSearchCV` and macro F1-score.
7. **Final evaluation** – Evaluate the selected model on the held-out grouped test partition and calculate classification metrics, confusion matrix, and one-vs-rest ROC-AUC.
8. **Serialization and deployment** – Save the full vectorizer-and-classifier pipeline with Joblib. The Streamlit app loads this saved pipeline for predictions; it does not train the model when it starts.

### Hyperparameter search

The Logistic Regression pipeline is tuned over:

| Parameter | Values tested |
|---|---|
| `tfidf__ngram_range` | `(1, 1)`, `(1, 2)` |
| `tfidf__min_df` | `1`, `2` |
| `clf__C` | `0.5`, `1.0`, `2.0` |

The selection metric is **macro-averaged F1-score**, which gives equal weight to each sentiment class.

## Model Evaluation

The evaluation report records the following results for the selected Logistic Regression pipeline:

| Metric | Result |
|---|---:|
| Held-out test accuracy | 1.0000 (100%) |
| Held-out macro precision | 1.0000 (100%) |
| Held-out macro recall | 1.0000 (100%) |
| Held-out macro F1-score | 1.0000 (100%) |
| Held-out macro one-vs-rest ROC-AUC | 1.0000 (100%) |
| Mean grouped 5-fold CV accuracy | 0.9964 (99.64%) |
| Mean grouped 5-fold CV macro F1-score | 0.9964 (99.64%) |

The selected Grid Search parameters recorded in the report are `C=0.5`, `ngram_range=(1, 1)`, and `min_df=1`. The grouped holdout contains 8,000 training rows and 2,000 test rows.

### Evaluation visualizations

**Confusion matrix**

![Confusion matrix](outputs/confusion_matrix.png)

**Model comparison**

![Model comparison](outputs/model_comparison.png)

**Multiclass ROC curves (one-vs-rest)**

![ROC curves](outputs/roc_curves.png)

> **How to interpret these results:** the perfect held-out scores should not be interpreted as 100% expected accuracy on new, real-world reviews. The corpus contains only 300 distinct review texts, so the language patterns are unusually repetitive and lexically separable. Group-based splitting prevents the same exact review text from crossing the train/test boundary, but the small number of unique texts still limits the strength of the generalization claim. The results describe performance on this supplied dataset only.

## Run the Project Locally

### 1. Clone the repository

Replace the placeholder URL with the URL of your GitHub repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Alternatively, download the repository as a ZIP file and extract it.

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Streamlit application

The repository includes a trained model at `models/sentiment_pipeline.joblib`, so the app can be started without retraining:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) in the terminal. Open that address in a browser.

## Use the Streamlit Application

1. Start the app using `streamlit run app.py`.
2. Enter or paste a customer review into the text area.
3. Select **Analyze sentiment**.
4. View the predicted class and the model's estimated probability for each class.

The displayed probability is the classifier's estimate, not a guarantee that the prediction is correct. The app also explains the dataset limitation in its **About this model** section.

## Retrain the Model

To rebuild the model and regenerate the evaluation outputs from the CSV:

```bash
python train_model.py
```

The script expects the dataset at `data/reviews.csv`. It writes the serialized model to `models/sentiment_pipeline.joblib` and generates evaluation artifacts under `outputs/` and `reports/`.

The custom module `sentiment_utils.py` must remain in the project root. The saved scikit-learn pipeline references its `clean_text` function, so the module must be importable when the model is loaded by `app.py`.

## Project Outputs

| File | Description |
|---|---|
| `outputs/dataset_summary.json` | Dataset profile and cleaning summary |
| `outputs/sentiment_distribution.png` | Sentiment-class distribution |
| `outputs/review_length_distribution.png` | Review-length distribution by sentiment |
| `outputs/category_distribution.png` | Record counts by category |
| `outputs/source_distribution.png` | Record counts by feedback source |
| `outputs/model_comparison.csv` | Baseline model metrics |
| `outputs/evaluation_metrics.json` | Holdout and cross-validation metrics |
| `outputs/classification_report.txt` | Per-class precision, recall, and F1-score |
| `outputs/confusion_matrix.png` | Confusion matrix for the selected model |
| `outputs/roc_curves.png` | One-vs-rest ROC curves |
| `outputs/model_comparison.png` | Model metric comparison chart |
| `outputs/fit_diagnostic.json` | Training-versus-validation score diagnostic |
| `outputs/example_predictions.csv` | Example inference results |
| `outputs/test_predictions.csv` | Held-out test labels and predictions |
| `models/sentiment_pipeline.joblib` | Complete trained inference pipeline |
| `reports/evaluation_report.pdf` | Project evaluation report |
| `reports/evaluation_report.docx` | Editable evaluation report |

## Limitations and Future Improvements

### Current limitations

- The dataset has only 300 distinct review texts across 10,000 records.
- Repeated and templated text limits the diversity of language represented during training.
- Performance on sarcasm, spelling errors, mixed sentiment, slang, and unfamiliar domains has not been established.
- Model probability estimates are not necessarily calibrated confidence values.
- The project is an academic prototype and has not been validated for production decisions.

### Future improvements

- Collect a larger dataset containing independently authored reviews.
- Verify sentiment labels and document the annotation process.
- Add examples containing spelling variations, sarcasm, mixed sentiment, and domain-specific vocabulary.
- Evaluate on a separate external dataset that was not used for model selection.
- Investigate probability calibration and monitor performance after deployment.
- Extend the application with batch CSV prediction and downloadable results.

## Learning Outcomes

This project demonstrates how to:

- Prepare and inspect a text dataset using Pandas.
- Perform exploratory data analysis and interpret visualizations.
- Apply text normalization without discarding important negation terms.
- Explain the difference between Bag of Words and TF-IDF.
- Train and compare supervised text-classification models.
- Evaluate a multiclass classifier using standard metrics and diagnostic plots.
- Use group-aware validation to reduce leakage from repeated text.
- Tune hyperparameters with Grid Search and cross-validation.
- Serialize a complete machine-learning pipeline and deploy it with Streamlit.

---

**Academic project:** Month 2 – AI-Powered Sentiment Analysis System  
**Focus:** Natural Language Processing, Machine Learning, and Streamlit deployment
