# Quora Duplicate Question Classifier

An NLP-based machine learning application that predicts whether two questions are **duplicate questions** (semantically asking the same thing).

The project uses a combination of **handcrafted NLP features, Word2Vec embeddings, and XGBoost** to classify question pairs. The final model is calibrated using **Isotonic Regression** to improve the reliability of predicted probabilities.

## Features

* Accepts two questions as input
* Performs text preprocessing
* Extracts handcrafted NLP features
* Generates Word2Vec sentence representations
* Uses XGBoost for classification
* Uses isotonic calibration for improved probability estimates
* Provides duplicate / not-duplicate predictions through a Streamlit interface

## Machine Learning Pipeline

```text
Question 1 ──┐
             ├── Text Preprocessing
Question 2 ──┘
                    │
                    ▼
            Feature Engineering
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
   Handcrafted NLP       Word2Vec Features
      Features
          │                   │
          └─────────┬─────────┘
                    ▼
               XGBoost
                    │
                    ▼
          Isotonic Calibration
                    │
                    ▼
             Final Prediction
```

## Feature Engineering

### 1. Basic Features

The model uses basic question-level and word-overlap features such as:

* Question length
* Number of words
* Common words
* Total unique words
* Common-word / total-word ratio

### 2. Token Features

Additional token-level features include:

* Common non-stopword ratios
* Common stopword ratios
* Common token ratios
* First word matching
* Last word matching

### 3. Length Features

The model also considers:

* Absolute difference in question lengths
* Average question length
* Longest common substring ratio

### 4. Fuzzy Matching Features

Four fuzzy string similarity features are used:

* Fuzzy ratio
* Partial ratio
* Token sort ratio
* Token set ratio

### 5. Word2Vec Features

Word2Vec is used to generate a 100-dimensional representation for each question.

For each question pair, the model uses:

* Question 1 Word2Vec vector — 100 features
* Question 2 Word2Vec vector — 100 features
* Absolute difference between the vectors — 100 features
* Element-wise product of the vectors — 100 features

This gives **400 Word2Vec-based features**.

Combined with the **22 handcrafted features**, the final model uses **422 features**.

## Model

The final classifier is:

**XGBoost + Isotonic Calibration**

The XGBoost classifier was trained with:

```python
n_estimators=300
max_depth=6
learning_rate=0.05
subsample=0.8
colsample_bytree=0.8
```

Isotonic calibration was then applied using 5-fold cross-validation:

```python
CalibratedClassifierCV(
    xgb,
    method="isotonic",
    cv=5
)
```

## Results

On the project's 40,000-question-pair dataset and evaluation split:

| Model                              |   Accuracy |  Log Loss |
| ---------------------------------- | ---------: | --------: |
| Random Forest                      |     78.14% |     0.448 |
| XGBoost                            |     80.08% |     0.404 |
| XGBoost + Sigmoid Calibration      |     80.38% |     0.410 |
| **XGBoost + Isotonic Calibration** | **80.46%** | **0.403** |

The calibrated XGBoost model achieved approximately **80.46% accuracy** with a log loss of approximately **0.403** on this evaluation setup.

> These results are specific to the dataset subset, feature engineering, train/test split, and preprocessing used in this project and should not be interpreted as a Kaggle leaderboard score.

## Tech Stack

### Programming

* Python

### NLP & Machine Learning

* NumPy
* scikit-learn
* XGBoost
* Gensim
* NLTK
* FuzzyWuzzy
* Distance

### Web Application

* Streamlit

### Data / Text Processing

* BeautifulSoup
* Regular Expressions

## Project Structure

```text
duplicate-question-classifier/
│
├── app.py                  # Streamlit application
├── helper.py               # Preprocessing and feature engineering
├── model.pkl               # Trained calibrated XGBoost model
├── w2v.pkl                 # Trained Word2Vec model
├── stopwords.pkl           # Saved English stopwords
├── create_stopwords.py     # Script for creating stopwords.pkl
├── requirements.txt        # Python dependencies
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/NoumanParvez12/duplicate-question-classifier.git
cd duplicate-question-classifier
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If NLTK stopwords need to be downloaded:

```python
import nltk
nltk.download('stopwords')
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Enter two questions and click **Find** to receive the prediction.

## Example

### Input

```text
Question 1:
How can I learn Python?

Question 2:
What is the best way to start learning Python?
```

### Output

```text
Duplicate
```

The model determines this based on the learned representation and engineered similarity features rather than relying only on exact string matching.

## Why Word2Vec?

Traditional Bag-of-Words representations primarily capture word occurrence and can produce very high-dimensional sparse vectors.

Word2Vec provides dense vector representations where words with similar contexts can have similar representations.

In this project, the word vectors are averaged to create a fixed-size representation of each question.

The model then compares the two question representations using their:

* Individual embeddings
* Absolute vector difference
* Element-wise vector product

## Future Improvements

* Experiment with transformer-based sentence embeddings
* Use pretrained models such as Sentence-BERT
* Add a larger training dataset
* Perform systematic hyperparameter tuning
* Improve handling of out-of-vocabulary words
* Add probability/confidence display in the Streamlit UI
* Deploy the application as a public web application
* Compare Word2Vec against TF-IDF and transformer embeddings

## Author

**Nouman Parvez**

GitHub:
https://github.com/NoumanParvez12

---

⭐ If you find this project useful, consider giving the repository a star.
