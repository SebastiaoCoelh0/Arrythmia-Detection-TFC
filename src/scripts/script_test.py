# import train models
import os
import pandas as pd
from src.processing.load_wfdb_data import load_raw_wfdb_data, enrich_metadata
from src.processing.rr_features import extract_peaks_find_peaks, extract_peaks_neurokit, extract_peaks_pan_tompkins, \
    filter_peaks, add_rr_metrics_to_df, add_diagnosis_column
from src.model.train_models import split_features_target, model_weight_balanced, model_undersampling


def main():
    # Load raw WFDB data
    df_ecg = load_raw_wfdb_data(base_path="C:/Users/sebas/PycharmProjects/TFC/data/WFDBRecords")
    df_ecg = enrich_metadata(df_ecg, base_path="C:/Users/sebas/PycharmProjects/TFC/data/WFDBRecords")

    # Extract R-peaks using different methods
    df_ecg_fp = extract_peaks_find_peaks(df_ecg)
    df_ecg_nk = extract_peaks_neurokit(df_ecg)
    df_ecg_pt = extract_peaks_pan_tompkins(df_ecg)

    # Filter peaks to keep only those with a reasonable number of R-peaks
    df_ecg_fp = filter_peaks(df_ecg_fp, min_peaks=6, max_peaks=10)
    df_ecg_nk = filter_peaks(df_ecg_nk, min_peaks=6, max_peaks=10)
    df_ecg_pt = filter_peaks(df_ecg_pt, min_peaks=6, max_peaks=10)

    # Add RR metrics to each DataFrame
    df_ecg_fp = add_rr_metrics_to_df(df_ecg_fp)
    df_ecg_nk = add_rr_metrics_to_df(df_ecg_nk)
    df_ecg_pt = add_rr_metrics_to_df(df_ecg_pt)

    # Add diagnosis column to each DataFrame
    df_ecg_fp = add_diagnosis_column(df_ecg_fp)
    df_ecg_nk = add_diagnosis_column(df_ecg_nk)
    df_ecg_pt = add_diagnosis_column(df_ecg_pt)

    # Split features and target
    X_train_fp, X_test_fp, y_train_fp, y_test_fp = split_features_target(df_ecg_fp)
    X_train_nk, X_test_nk, y_train_nk, y_test_nk = split_features_target(df_ecg_nk)
    X_train_pt, X_test_pt, y_train_pt, y_test_pt = split_features_target(df_ecg_pt)

    # Train models with balanced weights
    model_weight_balanced(X_train_fp, y_train_fp, X_test_fp, y_test_fp, df_name="find_peaks")
    model_weight_balanced(X_train_nk, y_train_nk, X_test_nk, y_test_nk, df_name="neurokit")
    model_weight_balanced(X_train_pt, y_train_pt, X_test_pt, y_test_pt, df_name="pan_tompkins")

    # Train models with undersampling
    model_undersampling(X_train_fp, y_train_fp, X_test_fp, y_test_fp, df_name="find_peaks")
    model_undersampling(X_train_nk, y_train_nk, X_test_nk, y_test_nk, df_name="neurokit")
    model_undersampling(X_train_pt, y_train_pt, X_test_pt, y_test_pt, df_name="pan_tompkins")


if __name__ == "__main__":
    main()
