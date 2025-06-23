from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import neurokit2 as nk
from systole.detectors import pan_tompkins
import matplotlib.pyplot as plt
import warnings


def extract_peaks_find_peaks(df, fs=500):
    # Extract R-peaks using the scipy find_peaks method and add a 'r_peaks' column to the DataFrame.
    df_fp = df.copy()
    r_peaks_list = []
    for signal in df_fp["signal"]:
        peaks, _ = find_peaks(signal, distance=fs * 0.3, prominence=0.4)
        r_peaks_list.append(peaks.tolist())

    df_fp["r_peaks"] = r_peaks_list

    mean_peaks = np.mean([len(peaks) for peaks in r_peaks_list])
    print(f"R-peaks extracted with find_peaks | Mean peaks: {mean_peaks:.2f} | Mean BPM: {mean_peaks * 6:.2f}")

    return df_fp


def extract_peaks_neurokit(df, fs=500):
    # Extract R-peaks using neurokit2.ecg_peaks and add a 'r_peaks' column.
    print("\n" + "=" * 80)
    print("Extracting R-peaks using neurokit2.ecg_peaks\n...")
    df_nk = df.copy()
    r_peaks_list = []
    for signal in df_nk["signal"]:
        try:
            _, peaks_dict = nk.ecg_peaks(signal, sampling_rate=fs)
            r_peaks_list.append(peaks_dict["ECG_R_Peaks"].tolist())
        except Exception as e:
            print("Error with neurokit2.ecg_peaks:", e)
            r_peaks_list.append([])

    df_nk["r_peaks"] = r_peaks_list

    mean_peaks = np.mean([len(peaks) for peaks in r_peaks_list])
    print(f"R-peaks extracted with neurokit | Mean peaks: {mean_peaks:.2f} | Mean BPM: {mean_peaks * 6:.2f}")

    return df_nk


def extract_peaks_pan_tompkins(df, fs=500):
    # Extract R-peaks using the Pan-Tompkins algorithm and add a 'r_peaks' column.
    df_pt = df.copy()
    r_peaks_list = []
    for signal in df_pt["signal"]:
        try:
            peaks = pan_tompkins(signal, sfreq=fs)
            r_peaks_list.append(peaks.tolist())
        except Exception as e:
            print("Error with pan_tompkins:", e)
            r_peaks_list.append([])

    df_pt["r_peaks"] = r_peaks_list

    mean_peaks = np.mean([len(peaks) for peaks in r_peaks_list])
    print(f"R-peaks extracted with pan tomplins | Mean peaks: {mean_peaks:.2f} | Mean BPM: {mean_peaks * 6:.2f}")

    return df_pt


def filter_peaks(df, min_peaks=6.7, max_peaks=20):
    # Filter DF to keep only records within min to max R-peaks
    # Defaults are set to 6.7 and 20 because the values for a bpm should be between 60 and 100
    # But a margin of 20 is added to avoid losing records
    print("\n" + "=" * 80)
    print("Filtering ECG Records Based on number of R-Peaks")
    bpm_min = min_peaks * 6
    bpm_max = max_peaks * 6
    total_before = len(df)

    df_filtrado = df[df["r_peaks"].apply(lambda x: min_peaks <= len(x) <= max_peaks)].copy()

    total_after = len(df_filtrado)
    print(f"Filtered by BPM ({bpm_min:.0f}-{bpm_max:.0f}): {total_before} ➝ {total_after} records")

    return df_filtrado


def add_diagnosis_column(df, target_diagnoses=None, new_column="has_diagnosis"):
    # Adds a binary column to the DataFrame indicating whether any of the target diagnoses are present.
    # target_diagnoses: list of acronyms to look for (e.g., ["AFIB", "APB"]). If None, defaults to ["AFIB"].
    # new_column: name of the new column to be added

    print("\n" + "=" * 80)
    print("   Adding Diagnosis Column to DataFrame")

    if target_diagnoses is None:
        target_diagnoses = ["AFIB"]

    df[new_column] = df["diagnosticos"].apply(
        lambda diagnoses: int(any(d in diagnoses for d in target_diagnoses))
    )

    total_cases = len(df)
    positive_cases = df[new_column].sum()
    negative_cases = total_cases - positive_cases
    print(f"Total cases: {total_cases}, Positive cases: {positive_cases}, Negative cases: {negative_cases}")
    print(f"Percentage of positive cases: {positive_cases / total_cases * 100:.2f}%")
    print(f"Added column '{new_column}' with {len(target_diagnoses)} diagnoses")

    return df


def add_rr_metrics_to_df(df, fs=500):
    # Calculate basic HRV metrics from the 'r_peaks' column and add them to the DataFrame.

    print("\n" + "=" * 80)
    print("Calculating RR intervals and HRV metrics\n...")
    metrics_list = []

    for r_peaks in df["r_peaks"]:
        if len(r_peaks) < 2:
            metrics_list.append({
                "rr_std": None,
                "rr_mean": None,
                "rr_cv": None,
                "pnn50": None,
                "rmssd": None,
                "skewness": None,
                "kurtosis": None
            })
            continue

        rr = np.diff(r_peaks) / fs
        rr_std = np.std(rr)
        rr_mean = np.mean(rr)
        rr_cv = rr_std / rr_mean if rr_mean > 0 else None
        pnn50 = np.sum(np.abs(np.diff(rr)) > 0.05) / len(rr)
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            skewness_val = skew(rr) if len(rr) >= 3 else None
            kurtosis_val = kurtosis(rr) if len(rr) >= 3 else None

        metrics_list.append({
            "rr_std": rr_std,
            "rr_mean": rr_mean,
            "rr_cv": rr_cv,
            "pnn50": pnn50,
            "rmssd": rmssd,
            "skewness": skewness_val,
            "kurtosis": kurtosis_val
        })

    df_metrics = pd.DataFrame(metrics_list)
    df = pd.concat([df.reset_index(drop=True), df_metrics], axis=1)

    print("Added RR metrics to DataFrame")

    return df


def plot_signals_with_r_peaks(df, fs=500, num_signals=5):
    # Plots ECG signals with precomputed R-peaks from the 'r_peaks' column.

    for i in range(min(num_signals, len(df))):
        signal = df.iloc[i]["signal"]
        r_peaks = df.iloc[i]["r_peaks"]
        time = np.arange(len(signal)) / fs
        diagnosis = ", ".join(df.iloc[i]["diagnosticos"])
        record_id = df.iloc[i]["record_id"]

        # Plot
        plt.figure(figsize=(12, 3))
        plt.plot(time, signal, label=f"Record {record_id}")
        plt.plot(time[r_peaks], np.array(signal)[r_peaks], "rx", label="R-peaks")
        plt.title(
            f"Record: {record_id} | Diagnosis: {diagnosis} | Number of R-peaks: {len(r_peaks)} ({len(r_peaks) * 6} BPM)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
