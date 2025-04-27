import pandas as pd
from systole.detectors import pan_tompkins
from functions import rr_metrics, test_models_fun
import os

print("Começar a correr o script")
df = pd.read_pickle("../data/processed/df_data.pkl")
print("df lido com sucesso:", df.shape)

fs = 500

# Lista para armazenar os picos R
r_peaks_list = []

# Só Pan-Tompkins
for i in range(len(df)):
    sinal = df.iloc[i]["signal"]
    r_peaks = pan_tompkins(sinal, sfreq=fs).tolist()
    r_peaks_list.append(r_peaks)

# Adicionar os r_peaks ao DataFrame
df["r_peaks"] = r_peaks_list
print("r_peaks adicionados com sucesso")

# Calcular as métricas RR (com métricas complexas)
metrics_list = []

for i in range(len(df)):
    metrics = rr_metrics(df.iloc[i]["r_peaks"], fs=fs, metricas_complexas=True)
    metrics_list.append(metrics)

# Concatenar as métricas no DataFrame
# Remover None da lista de métricas
valid_indices = [i for i, m in enumerate(metrics_list) if m is not None]
df_valid = df.iloc[valid_indices].copy()
metrics_valid = [metrics_list[i] for i in valid_indices]

# Concatenar só os válidos
df = pd.concat([df_valid.reset_index(drop=True), pd.DataFrame(metrics_valid)], axis=1)
print("rr_metrics adicionados com sucesso")

# Filtrar casos com 4 a 23 picos R
df_filtrado = df[df["r_peaks"].apply(lambda x: 4 <= len(x) <= 23)].copy()
print("df filtrado com sucesso:", df_filtrado.shape)

# Criar a variável target afib
df_ml = df_filtrado[[
    "rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis",
    "sample_entropy", "sd1", "sd2", "diagnosticos"
]].copy()

df_ml["afib"] = df_ml["diagnosticos"].apply(lambda x: "AFIB" in x).astype(int)

print("Casos com AFIB:", df_ml["afib"].sum())
print("Casos sem AFIB:", len(df_ml) - df_ml["afib"].sum())

# Criar pasta "df_ml" se ainda não existir
os.makedirs("../df_ml", exist_ok=True)

# Guardar como pickle
df_ml.to_pickle("df_ml/df_ml_complete.pkl")
print("df_ml_complete.pkl guardado com sucesso!")
