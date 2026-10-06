import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


DATA_FILE = "emails.csv"


def create_demo_dataset():
    """Create a small educational dataset if emails.csv is not present."""
    phishing = [
        "URGENT: Your account will be suspended. Verify your password immediately at http://secure-account.example",
        "Congratulations! You won a prize. Click here to claim your reward now.",
        "Your bank account has been locked. Confirm your login details using the link below.",
        "Security alert: unusual activity detected. Verify your identity within 24 hours.",
        "Your email storage is full. Sign in now to prevent account closure.",
        "You have an unpaid invoice. Open the attached link and confirm your payment information.",
        "Important notice: your account needs verification. Enter your username and password now.",
        "Final warning: your subscription will expire today. Click the link to renew.",
        "We detected suspicious activity. Confirm your card number to secure your account.",
        "You have been selected for a cash reward. Send your details to receive the money.",
        "Your mailbox will be deleted unless you verify your account immediately.",
        "Reset your password now by clicking this urgent verification link.",
    ]

    safe = [
        "Meeting reminder: the project review is scheduled for tomorrow at 10 AM.",
        "Hi team, please find the weekly report attached for your review.",
        "Your order has been shipped and will arrive on Friday.",
        "Thank you for attending the training session. The slides are available on the company portal.",
        "Can we reschedule our meeting to Thursday afternoon?",
        "The library will be closed on Sunday and will reopen on Monday morning.",
        "Your monthly electricity bill is available in your customer account.",
        "Here are the notes from today's classroom discussion.",
        "Please submit your assignment before the deadline mentioned by your instructor.",
        "The company picnic is planned for next Saturday. Please confirm your attendance.",
        "Your appointment is confirmed for 3 PM on Tuesday.",
        "Attached is the invoice for the services completed last month.",
    ]

    rows = [{"text": text, "label": "phishing"} for text in phishing]
    rows += [{"text": text, "label": "safe"} for text in safe]

    return pd.DataFrame(rows)


def load_dataset():
    """Load emails.csv or create the demo dataset."""
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded dataset: {DATA_FILE}")
    except FileNotFoundError:
        print(f"{DATA_FILE} not found. Using built-in demo dataset.")
        df = create_demo_dataset()

    # Accept a few common column names.
    text_column = next(
        (
            col
            for col in df.columns
            if col.lower() in {"text", "email", "email_text", "message", "body"}
        ),
        None,
    )

    label_column = next(
        (
            col
            for col in df.columns
            if col.lower() in {"label", "class", "category", "target"}
        ),
        None,
    )

    if text_column is None or label_column is None:
        raise ValueError(
            "Dataset must contain a text column (text/email/message/body) "
            "and a label column (label/class/category/target)."
        )

    df = df[[text_column, label_column]].dropna()
    df.columns = ["text", "label"]

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # Normalize common labels.
    label_map = {
        "1": "phishing",
        "0": "safe",
        "spam": "phishing",
        "ham": "safe",
        "legitimate": "safe",
        "phish": "phishing",
    }
    df["label"] = df["label"].replace(label_map)

    valid = df["label"].isin(["phishing", "safe"])
    df = df[valid]

    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both phishing and safe examples.")

    return df


def extract_url_features(text):
    """Return simple URL-related features for one email."""
    urls = re.findall(r"https?://\S+|www\.\S+", text, flags=re.IGNORECASE)

    suspicious_words = [
        "verify",
        "urgent",
        "password",
        "login",
        "confirm",
        "account",
        "click",
        "suspended",
        "reward",
        "payment",
    ]

    lower_text = text.lower()

    return {
        "url_count": len(urls),
        "suspicious_keyword_count": sum(
            word in lower_text for word in suspicious_words
        ),
        "has_ip_url": bool(
            re.search(
                r"https?://(?:\d{1,3}\.){3}\d{1,3}",
                text,
                flags=re.IGNORECASE,
            )
        ),
    }


def build_model():
    """TF-IDF + Logistic Regression text classifier."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=10000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )


def print_feature_analysis(text):
    features = extract_url_features(text)

    print("\n--- Email Feature Analysis ---")
    print(f"URLs found                 : {features['url_count']}")
    print(f"Suspicious keyword count   : {features['suspicious_keyword_count']}")
    print(
        f"URL contains an IP address: "
        f"{'Yes' if features['has_ip_url'] else 'No'}"
    )


def main():
    print("=" * 60)
    print("             PHISHING EMAIL DETECTION MODEL")
    print("=" * 60)

    df = load_dataset()

    print(f"\nTotal emails: {len(df)}")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, zero_division=0))

    cm = confusion_matrix(y_test, predictions, labels=["safe", "phishing"])

    print("\n--- Confusion Matrix ---")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Safe", "Phishing"],
    )
    disp.plot()
    plt.title("Phishing Email Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    print("\nConfusion matrix saved as: confusion_matrix.png")

    print("\n--- Test Your Own Email ---")
    email = input("Paste an email/message (or press Enter to skip): ").strip()

    if email:
        prediction = model.predict([email])[0]
        probabilities = model.predict_proba([email])[0]
        classes = model.classes_

        confidence = probabilities[list(classes).index(prediction)]

        print_feature_analysis(email)
        print(f"\nPrediction : {prediction.upper()}")
        print(f"Confidence : {confidence * 100:.2f}%")

        if prediction == "phishing":
            print("Warning: Treat this message carefully and verify the sender.")
        else:
            print("The model classified this message as Safe.")

    print("\nDone.")


if __name__ == "__main__":
    main()
