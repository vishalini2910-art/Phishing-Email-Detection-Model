# Phishing Email Detection Model

A beginner-friendly machine learning project that classifies email text as **Phishing** or **Safe** using Scikit-learn.

## Features

- Train a machine learning model on phishing and legitimate email text.
- Convert email text into TF-IDF features.
- Analyze simple URL-related features such as:
  - Number of URLs
  - Suspicious keywords
  - Whether a URL contains an IP address
- Classify new email text as `Phishing` or `Safe`.
- Display model accuracy.
- Display a classification report.
- Generate and display a confusion matrix.
- Allow you to test your own email text.
- Includes a small built-in demo dataset, so the project can run even without downloading a dataset.

## Project Structure

```text
phishing_email_detection_model/
│
├── phishing_detector.py
├── emails.csv                 # optional real dataset
├── requirements.txt
└── README.md
```

After running the program:

```text
confusion_matrix.png
```

will also be generated.

## Requirements

- Python 3.9 or newer
- pip

Install the required packages:

```bash
pip install -r requirements.txt
```

If `pip` does not work on Windows:

```bash
py -m pip install -r requirements.txt
```

## How to Run

Open the project folder in VS Code and run:

```bash
python phishing_detector.py
```

On Windows:

```bash
py phishing_detector.py
```

If `emails.csv` is not present, the program automatically uses its built-in educational dataset.

## Dataset Format

For training with your own dataset, create a file named:

```text
emails.csv
```

The simplest format is:

```csv
text,label
"Your account is suspended. Verify your password now.",phishing
"Please find the meeting notes attached.",safe
"Click this link immediately to claim your reward.",phishing
"Your order has been shipped and will arrive tomorrow.",safe
```

The program accepts common text-column names such as:

- `text`
- `email`
- `email_text`
- `message`
- `body`

It accepts common label-column names such as:

- `label`
- `class`
- `category`
- `target`

Common labels such as `spam`, `ham`, `legitimate`, `1`, and `0` are also normalized.

## How the Machine Learning Model Works

### 1. Dataset

The model receives examples of phishing and legitimate emails.

### 2. TF-IDF

`TfidfVectorizer` converts email text into numerical features.

TF-IDF gives higher importance to words that are useful for distinguishing documents while reducing the influence of very common words.

The project also uses:

```text
ngram_range=(1, 2)
```

so it can learn both individual words and two-word combinations.

### 3. Classification

The project uses:

```text
LogisticRegression
```

to classify the TF-IDF features into:

```text
Phishing
Safe
```

### 4. URL and Keyword Analysis

The program also performs a simple rule-based feature analysis on a supplied email.

It checks:

- URL count
- Suspicious keywords
- IP-address-based URLs

These features are displayed to the user as supporting information.

The main classifier in this beginner version is text-based TF-IDF + Logistic Regression.

## Accuracy

The program prints:

```text
Accuracy: XX.XX%
```

It also displays a classification report containing:

- Precision
- Recall
- F1-score
- Support

Do not expect the tiny built-in demo dataset to produce meaningful real-world accuracy. For a proper project, use a sufficiently large, representative, correctly labeled dataset and evaluate it on unseen test data.

## Confusion Matrix

The confusion matrix shows:

```text
                 Predicted
               Safe  Phishing

Actual Safe
Actual Phishing
```

It helps identify false positives and false negatives.

For a phishing detector, false negatives are particularly important because they represent phishing emails incorrectly classified as safe.

## Example Output

```text
============================================================
             PHISHING EMAIL DETECTION MODEL
============================================================

Total emails: 24

safe        12
phishing    12

Accuracy: XX.XX%

--- Classification Report ---
...

--- Confusion Matrix ---
...

Confusion matrix saved as: confusion_matrix.png

--- Test Your Own Email ---
Paste an email/message (or press Enter to skip):
```

## Important Note About Real-World Security

This project is an educational machine-learning demonstration. A real email security system should use many additional signals, including sender authentication, domain reputation, URL reputation, attachment analysis, email headers, and continuously updated threat intelligence.

A machine-learning prediction should not be treated as proof that an email is safe.

## Learning Outcomes

After completing this project, you should understand:

- Machine learning classification
- Training and testing datasets
- Train/test split
- TF-IDF
- Logistic Regression
- Text classification
- URL feature extraction
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

## GitHub Upload

After testing:

```bash
git init
git add .
git commit -m "Add phishing email detection model"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your own GitHub repository URL.

## Internship / Submission Explanation

You can explain the project like this:

> "I developed a machine-learning based phishing email detection model using Python and Scikit-learn. The system uses TF-IDF to convert email text into numerical features and Logistic Regression to classify messages as phishing or safe. I also added basic URL and suspicious-keyword analysis and evaluated the model using accuracy, a classification report, and a confusion matrix."

## Disclaimer

This project is intended for educational and cybersecurity awareness purposes. It should not be used as the sole security control for real email systems.
