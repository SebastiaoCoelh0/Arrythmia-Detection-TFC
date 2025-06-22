from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt


def df_info(df):
    # Print basic information about the DataFrame.
    if df.empty:
        print("DataFrame is empty.")
        return

    print("DataFrame Loaded Successfully")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Total records: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")


def split_features_target(df, target_column="has_diagnosis", feature_columns=None, test_size=0.2, random_state=42):
    # Split selected features and target column from the DataFrame into train/test sets.

    if feature_columns is None:
        feature_columns = ["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis"]
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, model_name="", df_name=""):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name} - {df_name}")
    plt.show()


def clean_data(X, y):
    X = X.dropna()
    y = y.loc[X.index]
    X = X[~X.isin([np.inf, -np.inf]).any(axis=1)]
    y = y.loc[X.index]
    return X, y


def train_and_select_best_model(models, X_train, y_train, X_test, y_test, df_name, save=False, prefix="model"):
    best_model = None
    best_name = ""
    best_f1 = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        print(f"Model: {name} | F1-score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred, model_name=name, df_name=df_name)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    if best_model is not None:
        print(f"\n- Best model: {best_name}")
        print(f"- Best F1-score: {best_f1:.4f}")

        if save:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(root_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(best_model, os.path.join(models_dir, f"{prefix}_{best_name.lower()}_f1_{best_f1:.2f}.joblib"))
            print(f"- Saved best model: {best_name} (F1: {best_f1:.4f})")


def print_class_distribution(y, label=""):
    print(f"Class Distribution {f'- {label}' if label else ''}")
    print(f"- Total samples: {len(y)}")
    print(f"- Positive cases (diagnosed): {(y == 1).sum()} ({(y == 1).mean() * 100:.2f}%)")
    print(f"- Negative cases (no diagnosis): {(y == 0).sum()} ({(y == 0).mean() * 100:.2f}%)")


def model_weight_balanced(X_train, y_train, X_test, y_test, df_name="", save=False):
    print("\n" + "=" * 80)
    print("Balanced Model Training:")
    X_train, y_train = clean_data(X_train, y_train)
    X_test, y_test = clean_data(X_test, y_test)

    print_class_distribution(y=y_train, label="Balanced Model")
    print("=" * 80)

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    models = {
        "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "SVC": SVC(class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42)
    }

    train_and_select_best_model(models, X_train, y_train, X_test, y_test, df_name, save, prefix="balanced")


def model_undersampling(X_train, y_train, X_test, y_test, df_name="", save=False):
    print("\n" + "=" * 80)
    print("Under Sampling Model Training")
    X_train, y_train = clean_data(X_train, y_train)
    X_test, y_test = clean_data(X_test, y_test)
    print_class_distribution(y=y_train, label="Before Under Sampling")

    undersampler = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)

    print_class_distribution(y=y_train_res, label="After Under Sampling")
    print("=" * 80)

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVC": SVC(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    train_and_select_best_model(models, X_train_res, y_train_res, X_test, y_test, df_name, save, prefix="undersampling")


def model_default(X_train, y_train, X_test, y_test, df_name="", save=False):
    print("\n" + "=" * 80)
    print("Default Model Training:")
    X_train, y_train = clean_data(X_train, y_train)
    X_test, y_test = clean_data(X_test, y_test)

    print_class_distribution(y=y_train, label="Default Model")
    print("=" * 80)

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVC": SVC(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    train_and_select_best_model(models, X_train, y_train, X_test, y_test, df_name, save, prefix="default")
