from src.processing.load_wfdb_data import load_raw_wfdb_data, enrich_metadata
from src.processing.rr_features import extract_peaks_find_peaks, extract_peaks_neurokit, extract_peaks_pan_tompkins, \
    filter_peaks, add_rr_metrics_to_df, add_diagnosis_column
from src.model.train_models import split_features_target, model_weight_balanced, model_undersampling
import pandas as pd
import os


def main():
    # Load pkl
    df = pd.read_pickle(os.path.join("C:/Users/sebas/PycharmProjects/TFC/data/processed", "df_data.pkl"))
    # print(df.head())
    print(df.columns.tolist())
    print(f"Total records loaded: {len(df)}")
    # ['record_id', 'signal', 'age', 'sex', 'diagnosticos']
    # Total records loaded: 45150

    # Extract R-peaks using different methods
    # df_ecg_fp = extract_peaks_find_peaks(df)
    # df_ecg_nk = extract_peaks_neurokit(df)
    df_ecg_pt = extract_peaks_pan_tompkins(df)

    # Mean
    # print(f"Mean number of R-peaks (find_peaks): {df_ecg_fp['r_peaks'].apply(len).mean()}")
    # print(f"Mean number of R-peaks (neurokit): {df_ecg_nk['r_peaks'].apply(len).mean()}")
    print(f"Mean number of R-peaks (pan_tompkins): {df_ecg_pt['r_peaks'].apply(len).mean()}")
    # Filter peaks to keep only those with a reasonable number of R-peaks
    # print("Filtering peaks find_peaks")
    # df_ecg_fp = filter_peaks(df_ecg_fp)
    # print("Filtering peaks neurokit")
    # df_ecg_nk = filter_peaks(df_ecg_nk)
    print("Filtering peaks pan_tompkins")
    df_ecg_pt = filter_peaks(df_ecg_pt)
    # Add RR metrics to each DataFrame
    # df_ecg_fp = add_rr_metrics_to_df(df_ecg_fp)
    # df_ecg_nk = add_rr_metrics_to_df(df_ecg_nk)
    # df_ecg_pt = add_rr_metrics_to_df(df_ecg_pt)
    # # Add diagnosis column to each DataFrame
    # df_ecg_fp = add_diagnosis_column(df_ecg_fp)
    # df_ecg_nk = add_diagnosis_column(df_ecg_nk)
    # df_ecg_pt = add_diagnosis_column(df_ecg_pt)
    # # Split features and target
    # X_train_fp, X_test_fp, y_train_fp, y_test_fp = split_features_target(df_ecg_fp)
    # X_train_nk, X_test_nk, y_train_nk, y_test_nk = split_features_target(df_ecg_nk)
    # X_train_pt, X_test_pt, y_train_pt, y_test_pt = split_features_target(df_ecg_pt)
    # # Train models with balanced weights
    # model_weight_balanced(X_train_fp, y_train_fp, X_test_fp, y_test_fp, df_name="find_peaks")
    # model_weight_balanced(X_train_nk, y_train_nk, X_test_nk, y_test_nk, df_name="neurokit")
    # model_weight_balanced(X_train_pt, y_train_pt, X_test_pt, y_test_pt, df_name="pan_tompkins")
    # # Train models with undersampling
    # model_undersampling(X_train_fp, y_train_fp, X_test_fp, y_test_fp, df_name="find_peaks")
    # model_undersampling(X_train_nk, y_train_nk, X_test_nk, y_test_nk, df_name="neurokit")
    # model_undersampling(X_train_pt, y_train_pt, X_test_pt, y_test_pt, df_name="pan_tompkins")


if __name__ == "__main__":
    main()
