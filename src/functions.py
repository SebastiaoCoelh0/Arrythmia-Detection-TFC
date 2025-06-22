import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
import antropy as ant
import joblib
import os


def plot_confusion_matrix(y_true, y_pred, model_name="", df_name=""):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name} - {df_name}")
    plt.show()


def rr_metrics(r_peaks, fs=500, metricas_complexas=False):
    rr = np.diff(r_peaks) / fs

    if len(rr) < 2:
        return None

    metrics = {
        "rr_std": np.std(rr),
        "rr_mean": np.mean(rr),
        "rr_cv": np.std(rr) / np.mean(rr) if np.mean(rr) > 0 else None,
        "pnn50": np.sum(np.abs(np.diff(rr)) > 0.05) / len(rr) if len(rr) > 1 else None,
        "rmssd": np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else None,
        "skewness": skew(rr) if len(rr) >= 3 else None,
        "kurtosis": kurtosis(rr) if len(rr) >= 3 else None
    }

    if metricas_complexas:
        try:
            metrics.update({
                "sample_entropy": ant.sample_entropy(rr),
                "sd1": np.sqrt(np.std(np.diff(rr)) ** 2 / 2),
                "sd2": np.sqrt(2 * np.std(rr) ** 2 - np.std(np.diff(rr)) ** 2 / 2)
            })
        except Exception as e:
            # If there's an error in calculating complex metrics, set them to None
            metrics.update({
                "sample_entropy": None,
                "sd1": None,
                "sd2": None
            })

    return metrics


def test_models_fun(X_train, y_train, X_test, y_test, df_name="", save=False):
    print(f"Número de linhas antes de remover NaNs: {len(X_train)}")
    # Remover NaNs
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]
    # Remover inf ou valores absurdos
    X_train = X_train[~X_train.isin([np.inf, -np.inf]).any(axis=1)]
    y_train = y_train.loc[X_train.index]
    X_test = X_test[~X_test.isin([np.inf, -np.inf]).any(axis=1)]
    y_test = y_test.loc[X_test.index]

    print(f"Número de linhas depois de remover NaNs: {len(X_train)}")
    print("Casos sem AFIB:", (y_train == 0).sum())
    print("Casos com AFIB:", (y_train == 1).sum())

    scale_pos_weight_calc = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    models = {
        "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "SVC": SVC(class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight_calc, eval_metric='logloss',
                                 random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, model_name=name, df_name=df_name)
        print(f"Modelo: {name}")
        print(classification_report(y_test, y_pred))
        if save:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(root_dir, "models")
            # Guardar o modelo
            joblib.dump(model, os.path.join(models_dir, f"{name.lower()}_balances.joblib"))
            print(f"Modelo {name} guardado como {name.lower()}_balanced.joblib")


def test_undersampling_fun(X_train, y_train, X_test, y_test, df_name="", save=False):
    print(f"Número de linhas antes de remover NaNs: {len(X_train)}")

    # Remover NaNs
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]
    # Remover inf ou valores absurdos
    X_train = X_train[~X_train.isin([np.inf, -np.inf]).any(axis=1)]
    y_train = y_train.loc[X_train.index]
    X_test = X_test[~X_test.isin([np.inf, -np.inf]).any(axis=1)]
    y_test = y_test.loc[X_test.index]
    print(f"Número de linhas depois de remover NaNs: {len(X_train)}")

    print("Casos sem AFIB:", (y_train == 0).sum())
    print("Casos com AFIB:", (y_train == 1).sum())

    # Realizando o undersampling para equilibrar as classes
    undersampler = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)

    print(f"Número de linhas após o undersampling: {len(X_train_res)}")
    print("Casos sem AFIB após o undersampling:", (y_train_res == 0).sum())
    print("Casos com AFIB após o undersampling:", (y_train_res == 1).sum())

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVC": SVC(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)

        # Exibir a matriz de confusão
        plot_confusion_matrix(y_test, y_pred, model_name=name, df_name=df_name)

        print(f"Modelo: {name}")
        print(classification_report(y_test, y_pred))

        if save:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(root_dir, "models")
            # Guardar o modelo
            joblib.dump(model, os.path.join(models_dir, f"{name.lower()}_undersampling.joblib"))
            print(f"Modelo {name} guardado como {name.lower()}_undersampling.joblib")
