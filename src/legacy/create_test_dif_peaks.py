import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
import neurokit2 as nk
from systole.detectors import pan_tompkins
from functions import rr_metrics, test_models_fun

print("Começar a correr o script")
df1 = pd.read_pickle("../data/processed/df_data.pkl")
df2 = df1.copy()
df3 = df1.copy()

print("df lido com sucesso: ", df1.shape)

fs = 500
r_peaks_list1 = []
r_peaks_list2 = []
r_peaks_list3 = []

for i in range(len(df1)):
    sinal = df1.iloc[i]["signal"]
    # find_peaks
    picos1, _ = find_peaks(sinal, distance=fs * 0.3, prominence=0.4)
    r_peaks_list1.append(picos1.tolist())
    # nk.ecg_peaks
    _, picos2 = nk.ecg_peaks(sinal, sampling_rate=fs)
    r_peaks_list2.append(picos2["ECG_R_Peaks"].tolist())
    # pan_tompkins
    picos3 = pan_tompkins(sinal, sfreq=fs)
    r_peaks_list3.append(picos3.tolist())

df1["r_peaks"] = r_peaks_list1
df2["r_peaks"] = r_peaks_list2
df3["r_peaks"] = r_peaks_list3
print("r_peaks adicionados com sucesso")

metrics_list1 = []
metrics_list2 = []
metrics_list3 = []

for i in range(len(df1)):
    metrics_list1.append(rr_metrics(df1.iloc[i]["r_peaks"]))
    metrics_list2.append(rr_metrics(df2.iloc[i]["r_peaks"]))
    metrics_list3.append(rr_metrics(df3.iloc[i]["r_peaks"]))

df1 = pd.concat([df1, pd.DataFrame(metrics_list1)], axis=1)
df2 = pd.concat([df2, pd.DataFrame(metrics_list2)], axis=1)
df3 = pd.concat([df3, pd.DataFrame(metrics_list3)], axis=1)
print("rr_metrics adicionados com sucesso")

# Filtrar casos com menos de 4 picos R
df_filtrado1 = df1[df1["r_peaks"].apply(lambda x: len(x) >= 4)].copy()
df_filtrado2 = df2[df2["r_peaks"].apply(lambda x: len(x) >= 4)].copy()
df_filtrado3 = df3[df3["r_peaks"].apply(lambda x: len(x) >= 4)].copy()

# Filtrar casos com mais de 23 picos R
df_filtrado1 = df_filtrado1[df_filtrado1["r_peaks"].apply(lambda x: len(x) <= 23)].copy()
df_filtrado2 = df_filtrado2[df_filtrado2["r_peaks"].apply(lambda x: len(x) <= 23)].copy()
df_filtrado3 = df_filtrado3[df_filtrado3["r_peaks"].apply(lambda x: len(x) <= 23)].copy()
print("df filtrados com sucesso")

print(f"df1:\nNúmero de amostras pré filtros: {len(df1)}")
print(f"Número de amostras após filtros: {len(df_filtrado1)}")
print(f"df2:\nNúmero de amostras pré filtros: {len(df2)}")
print(f"Número de amostras após filtros: {len(df_filtrado2)}")
print(f"df3:\nNúmero de amostras pré filtros: {len(df3)}")
print(f"Número de amostras após filtros: {len(df_filtrado3)}")

df_ml1 = df1[["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis", "diagnosticos"]].copy()
df_ml1["afib"] = df_ml1["diagnosticos"].apply(lambda x: "AFIB" in x).astype(int)
df_ml2 = df2[["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis", "diagnosticos"]].copy()
df_ml2["afib"] = df_ml2["diagnosticos"].apply(lambda x: "AFIB" in x).astype(int)
df_ml3 = df3[["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis", "diagnosticos"]].copy()
df_ml3["afib"] = df_ml3["diagnosticos"].apply(lambda x: "AFIB" in x).astype(int)

print("df1:\nCasos com AFIB:", df_ml1["afib"].sum())
print("Casos sem AFIB:", len(df_ml1) - df_ml1["afib"].sum())
print("df2:\nCasos com AFIB:", df_ml2["afib"].sum())
print("Casos sem AFIB:", len(df_ml2) - df_ml2["afib"].sum())
print("df3:\nCasos com AFIB:", df_ml3["afib"].sum())
print("Casos sem AFIB:", len(df_ml3) - df_ml3["afib"].sum())

# Guardar os dataframes como pickle na pasta df_ml
df_ml1.to_pickle("df_ml/df_ml_fp.pkl")
df_ml2.to_pickle("df_ml/df_ml_nk.pkl")
df_ml3.to_pickle("df_ml/df_ml_pt.pkl")
print("df_ml guardados com sucesso")

X1 = df_ml1[["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis"]]
y1 = df_ml1["afib"]
X2 = df_ml2[["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis"]]
y2 = df_ml2["afib"]
X3 = df_ml3[["rr_std", "rr_cv", "pnn50", "rmssd", "skewness", "kurtosis"]]
y3 = df_ml3["afib"]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, stratify=y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, stratify=y2, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, stratify=y3, test_size=0.2, random_state=42)

print(
    "Modelos com df1 - Find Peaks:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train1, y_train1, X_test1, y_test1, df_name="Find Peaks")
print(
    "Modelos com df2 - nk.ecg_peaks:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train2, y_train2, X_test2, y_test2, df_name="nk.ecg_peaks")
print(
    "Modelos com df3 - Pan Tompkins:\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
test_models_fun(X_train3, y_train3, X_test3, y_test3, df_name="Pan Tompkins")
