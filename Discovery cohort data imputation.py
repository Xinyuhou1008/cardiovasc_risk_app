import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
import miceforest as mf

SPLIT_SEED = 42
IMPUTE_SEED = 42
TEST_SIZE = 0.2
IMPUTE_ITERS = 5

INPUT_PATH = r"E:\Jupter_workplace\data\discovery.csv"

TRAIN_RAW_OUT = "training_cohort_raw.csv"
TEST_RAW_OUT = "testing_cohort_raw.csv"
TRAIN_IMPUTED_OUT = "training_cohort_imputed.csv"
TEST_IMPUTED_OUT = "testing_cohort_imputed.csv"

EVENT_COL = "cdeath"
TIME_COL = "time"
ID_COLS = []

def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

def get_categorical_columns(df: pd.DataFrame):
    cols = []
    max_pos = min(50, df.shape[1])
    for i in range(1, max_pos):
        col = df.columns[i]
        if col not in [EVENT_COL, TIME_COL] and col not in ID_COLS:
            cols.append(col)
    return cols

def split_discovery_dataset(df: pd.DataFrame):
    for col in [EVENT_COL, TIME_COL]:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    if df[EVENT_COL].isna().any():
        raise ValueError(f"{EVENT_COL} has missing values.")

    stratify = None
    counts = df[EVENT_COL].value_counts(dropna=False)
    if df[EVENT_COL].nunique(dropna=True) >= 2 and counts.min() >= 2:
        stratify = df[EVENT_COL]
    else:
        warnings.warn(f"{EVENT_COL} cannot be used for stratification.")

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=SPLIT_SEED,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as e:
        warnings.warn(f"Stratified split failed: {e}. Falling back to random split.")
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=SPLIT_SEED,
            shuffle=True,
            stratify=None,
        )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def impute_with_miceforest(df: pd.DataFrame, categorical_cols):
    exclude_cols = [EVENT_COL, TIME_COL] + ID_COLS
    impute_cols = [c for c in df.columns if c not in exclude_cols]

    if not impute_cols:
        raise ValueError("No columns available for imputation.")

    work = df[impute_cols].copy()
    original_dtypes = work.dtypes.to_dict()

    object_like_cols = work.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_like_cols:
        work[col] = work[col].astype("category")

    categorical_cols_in_work = [c for c in categorical_cols if c in work.columns]
    for col in categorical_cols_in_work:
        if work[col].dtype.name != "category":
            work[col] = work[col].astype("category")

    if work.select_dtypes(include=["object"]).columns.tolist():
        raise TypeError("Unprocessed object columns remain.")

    kernel = mf.ImputationKernel(
        data=work,
        random_state=IMPUTE_SEED,
        save_all_iterations_data=False,
    )
    kernel.mice(IMPUTE_ITERS)
    imputed_work = kernel.complete_data(dataset=0, inplace=False)

    for col in categorical_cols_in_work:
        if pd.api.types.is_numeric_dtype(original_dtypes[col]):
            imputed_work[col] = pd.to_numeric(imputed_work[col], errors="coerce").round().astype("Int64")
        else:
            imputed_work[col] = imputed_work[col].astype("string")

    for col in object_like_cols:
        if col in imputed_work.columns and col not in categorical_cols_in_work:
            imputed_work[col] = imputed_work[col].astype("string")

    keep_part = df[exclude_cols].copy()
    imputed_df = pd.concat([imputed_work, keep_part], axis=1)
    return imputed_df[df.columns]

def main():
    df = load_data(INPUT_PATH)

    missing_cols = [c for c in [EVENT_COL, TIME_COL] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")

    categorical_cols = get_categorical_columns(df)

    train_df, test_df = split_discovery_dataset(df)

    train_df.to_csv(TRAIN_RAW_OUT, index=False)
    test_df.to_csv(TEST_RAW_OUT, index=False)

    train_imputed = impute_with_miceforest(train_df, categorical_cols)
    test_imputed = impute_with_miceforest(test_df, categorical_cols)

    train_imputed.to_csv(TRAIN_IMPUTED_OUT, index=False)
    test_imputed.to_csv(TEST_IMPUTED_OUT, index=False)

    print("Done.")
    print(f"Saved: {TRAIN_RAW_OUT}")
    print(f"Saved: {TEST_RAW_OUT}")
    print(f"Saved: {TRAIN_IMPUTED_OUT}")
    print(f"Saved: {TEST_IMPUTED_OUT}")

if __name__ == "__main__":
    main()