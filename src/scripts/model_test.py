from src.processing.rr_features import extract_peaks_find_peaks, extract_peaks_neurokit, extract_peaks_pan_tompkins, \
    filter_peaks, plot_signals_with_r_peaks, add_diagnosis_column, add_rr_metrics_to_df
from src.model.train_models import split_features_target, model_weight_balanced, model_undersampling, train_test_split, \
    df_info, model_default
import pandas as pd
import os


def main():
    # Load the data
    df = pd.read_pickle(os.path.join("C:/Users/sebas/PycharmProjects/TFC/data/processed", "df_data.pkl"))
    df_info(df)

    # Extract R-peaks using neurokit
    df = extract_peaks_neurokit(df)

    # Filter the signals with invalid R-peaks
    df = filter_peaks(df)

    # Add RR metrics to the DataFrame
    df = add_rr_metrics_to_df(df)

    # Define the target diagnoses
    df = add_diagnosis_column(df,
                              target_diagnoses=["AFIB", "AF", "SA", "APB", "VPB", "ABI", "VB", "JEB", "VEB", "JPT",
                                                "VET", "WAVN", "SAAWR"])
    # Train-test split
    X_train, X_test, y_train, y_test = split_features_target(df)

    # Train models default
    model_default(X_train, y_train, X_test, y_test, df_name="Classic Model Multiple Diagnoses", save=False)

    # Train models with balanced weights
    model_weight_balanced(X_train, y_train, X_test, y_test, df_name="Balanced Multiple Diagnoses", save=False)

    # Train models with undersampling
    model_undersampling(X_train, y_train, X_test, y_test, df_name="Undersampling Multiple Diagnoses", save=False)
    """
    df_2 = df.copy()
    df_3 = df.copy()
    df = add_diagnosis_column(df, target_diagnoses=["AFIB", "2AVB1", "APB", "VPB", "VEB", "VET", "SA", "JPT", "WAVN",
                                                    "SAAWR"])

    # Train-test split
    X_train, X_test, y_train, y_test = split_features_target(df)

    # Train models with balanced weights
    model_weight_balanced(X_train, y_train, X_test, y_test, df_name="Balanced Multiple Diagnoses - 1")

    # Train models with undersampling
    model_undersampling(X_train, y_train, X_test, y_test, df_name="Undersampling Multiple Diagnoses - 1")

    # test other diagnoses
    df_2 = add_diagnosis_column(df_2, target_diagnoses=["AFIB", "SA", "APB", "VPB", "2AVB1", "AF",
                                                        "JPT", "WAVN", "VET", "ABI"])
    # Train-test split
    X_train, X_test, y_train, y_test = split_features_target(df_2)

    # Train models with balanced weights
    model_weight_balanced(X_train, y_train, X_test, y_test, df_name="Balanced Multiple Diagnoses - 2")

    # Train models with undersampling
    model_undersampling(X_train, y_train, X_test, y_test, df_name="Undersampling Multiple Diagnoses - 2")
    """


if __name__ == "__main__":
    main()
