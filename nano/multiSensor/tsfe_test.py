import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from label_data import get_train_data
import joblib  # for saving to disk

def format_for_tsfresh(X):
    """
    Converts (n_samples, window_length, n_channels) → tsfresh-compatible long-form DataFrame
    """
    n_samples, window_len, n_channels = X.shape
    records = []

    for sample_id in range(n_samples):
        for ch in range(n_channels):
            for t in range(window_len):
                records.append({
                    "id": sample_id,
                    "time": t,
                    "channel": ch,
                    "value": X[sample_id, t, ch]
                })

    df = pd.DataFrame(records)
    df = df.pivot_table(index=["id", "time"], columns="channel", values="value")
    df.columns = [f"ch{c}" for c in df.columns]
    df.reset_index(inplace=True)
    return df

if __name__ == "__main__":
    print("Loading raw data...")
    X_raw, y = get_train_data()

    print("Formatting for tsfresh...")
    tsfresh_input = format_for_tsfresh(X_raw)

    print("Extracting features (EfficientFCParameters)...")
    extraction_settings = EfficientFCParameters()
    X_features = extract_features(
        tsfresh_input,
        column_id="id",
        column_sort="time",
        default_fc_parameters=extraction_settings,
        n_jobs=0  # or -1 for all cores
    )

    print("Imputing missing values...")
    impute(X_features)

    print("Selecting relevant features...")
    X_filtered = select_features(X_features, y)

    print(f"Selected {X_filtered.shape[1]} features")

    # Save to disk
    print("Saving features and labels to disk...")
    joblib.dump(X_filtered, "X_filtered.pkl")
    joblib.dump(y, "y_labels.pkl")

    print("Done!")
