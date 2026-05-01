# Spam Classifier App

An end-to-end machine learning pipeline for spam and phishing email detection. Built with Python and Streamlit, this app trains a Naive Bayes classifier on real email data using TF-IDF feature engineering, then lets you classify new emails in real time through an interactive web interface.

---

## Features

- **Interactive web UI** powered by Streamlit
- **Real-time email classification** — paste any email text and get an instant spam/ham prediction with a confidence score
- **TF-IDF vectorization** for converting raw email text into numerical features
- **Multinomial Naive Bayes** classifier trained on a labeled email dataset
- **Model evaluation metrics** displayed on every run: accuracy, precision, recall, and F1 score
- **Data visualizations** including a bar chart and pie chart of the spam vs. ham distribution in the dataset
- **Dataset preview** rendered directly in the app

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Web Framework | Streamlit |
| ML / Modeling | scikit-learn (TF-IDF, Naive Bayes, train/test split) |
| Data Handling | pandas |
| Visualization | matplotlib |

---

## Project Structure

```
SpamClassifierApp/
├── app.py          # Main Streamlit application
└── emails.csv      # Labeled email dataset (label, text)
```

---

## Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed. Then install the required packages:

```bash
pip install streamlit pandas scikit-learn matplotlib
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How It Works

1. **Data Loading** — `emails.csv` is loaded into a pandas DataFrame. The dataset contains two columns: `label` (spam or ham) and `text` (the email body).

2. **Feature Engineering** — Email text is transformed into a TF-IDF matrix, converting word frequency patterns into numerical vectors suitable for classification.

3. **Model Training** — An 80/20 train/test split is applied, and a Multinomial Naive Bayes model is trained on the vectorized training data.

4. **Evaluation** — Accuracy, precision, recall, and F1 score are computed on the held-out test set and displayed in the UI.

5. **Inference** — Users paste an email into the text area, click **Classify**, and receive a `SPAM` or `HAM` prediction along with a confidence percentage.

---

## Dataset Format

The app expects `emails.csv` in the following format (with a header row):

```
label,text
ham,"Hey, are we still on for lunch today?"
spam,"CONGRATULATIONS! You've been selected for a $1000 gift card..."
```

---

## Model Performance

Performance metrics are computed dynamically each time the app runs against the 20% test split. Typical results on a standard spam dataset:

| Metric | Description |
|---|---|
| Accuracy | Overall correct classification rate |
| Precision (Spam) | Of emails predicted spam, how many actually were |
| Recall (Spam) | Of actual spam emails, how many were caught |
| F1 Score (Spam) | Harmonic mean of precision and recall |

---

## Future Improvements

- Support for uploading custom datasets via the UI
- Additional classifier options (Logistic Regression, SVM, etc.)
- Email header and metadata feature extraction
- Persistent model saving and loading with `joblib`
- Deployment to Streamlit Cloud or Hugging Face Spaces

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Author

**eastonwag** — [GitHub Profile](https://github.com/eastonwag)
