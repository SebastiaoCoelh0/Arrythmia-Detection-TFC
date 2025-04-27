import pandas as pd
from sklearn.model_selection import train_test_split
from functions import test_models_fun, test_undersampling_fun

print("Começar a correr o script")
df = pd.read_pickle("../df_ml/df_ml_pt.pkl")
print("df lido com sucesso:", df.shape)

# Separar as features
X1 = df[["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis"]]
# X2 = df[["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis", "sample_entropy", "sd1", "sd2"]]
y = df["afib"]

# Separar treino/teste
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y, stratify=y, test_size=0.2, random_state=42)
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, stratify=y, test_size=0.2, random_=42)

# Testar com features básicas
print(
    "Undersampling - Features básicas (rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis):\n"
    "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_undersampling_fun(X_train1, y_train1, X_test1, y_test1, df_name="basic_features", save=True)

print(
    "Weight balanced - Features básicas (rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis):\n"
    "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train1, y_train1, X_test1, y_test1, df_name="basic_features", save=True)

# Testar com features avançadas
# print(
#     "Undersampling - Features completas (rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis, sample_entropy, sd1, sd2):\n"
#     "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
# test_undersampling_fun(X_train2, y_train2, X_test2, y_test2, df_name="Features completas")
#
# print(
#     "Weight balanced - Features completas (rr_std, rr_cv, pnn50, rmssd, skewness, kurtosis, sample_entropy, sd1, sd2):\n"
#     "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
# test_models_fun(X_train2, y_train2, X_test2, y_test2, df_name="Features completas")
