import pandas as pd
from sklearn.model_selection import train_test_split

from functions import test_models_fun

print("Começar a correr o script")
df = pd.read_pickle("data/processed/df_ml_pt.pkl")
print("df lido com sucesso: ", df.shape)
X1 = df[["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis"]]
X2 = df[["pnn50", "rmssd", "skewness", "kurtosis"]]
X3 = df[["rr_std", "rr_cv"]]
X4 = df[["pnn50", "rmssd"]]
X5 = df[["skewness", "kurtosis"]]
y = df["afib"]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y, stratify=y, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, stratify=y, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y, stratify=y, test_size=0.2, random_state=42)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y, stratify=y, test_size=0.2, random_state=42)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y, stratify=y, test_size=0.2, random_state=42)

print(
    "Modelos com X1 - rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train1, y_train1, X_test1, y_test1, df_name="rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis")
print(
    "Modelos com X2 - pnn50, rmssd, skewness, kurtosis:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train2, y_train2, X_test2, y_test2, df_name="pnn50, rmssd, skewness, kurtosis")
print(
    "Modelos com X3 - rr_std, rr_cv:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train3, y_train3, X_test3, y_test3, df_name="rr_std, rr_cv")
print(
    "Modelos com X4 - pnn50, rmssd:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train4, y_train4, X_test4, y_test4, df_name="pnn50, rmssd")
print(
    "Modelos com X5 - skewness, kurtosis:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train5, y_train5, X_test5, y_test5, df_name="skewness, kurtosis")
