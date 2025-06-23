import pandas as pd
from sklearn.model_selection import train_test_split
from functions import test_models_fun, test_undersampling_fun

print("Começar a correr o script")
df = pd.read_pickle("data/processed/df_ml_pt.pkl")
print("df lido com sucesso: ", df.shape)
X = df[["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis"]]
y = df["afib"]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print(
    "Undersampling - rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_undersampling_fun(X_train1, y_train1, X_test1, y_test1, df_name="rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis")

print(
    "Wheight balanced - rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train1, y_train1, X_test1, y_test1, df_name="rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis")
print(
    "Undersampling - rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis, sample_entropy, sd1, sd2:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_undersampling_fun(X_train1, y_train1, X_test1, y_test1, df_name="rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis")

print(
    "Wheight balanced - rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis, sample_entropy, sd1, sd2:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train1, y_train1, X_test1, y_test1, df_name="rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis")
