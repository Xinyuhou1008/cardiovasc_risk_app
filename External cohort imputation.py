import warnings
from pathlib import Path

import pandas as pd
import miceforest as mf

IMPUTE_SEED = 42
IMPUTE_ITERS = 5

INPUT_PATH = r"E:\Jupter_workplace\data\external_cohort.csv"
OUTPUT_PATH = r"E:\Jupter_workplace\data\external_cohort_imputed.csv"

EVENT_COL = "cdeath"

ORDINAL_RANGES = {
    "D_dimer": (0, 3),
    "BNP": (1, 4),
    "GLU": (0, 3),
    "TG": (1, 3),
    "cTnI": (1, 5),
    "LDL_C": (1, 7),
}

def load_data(path: str) -> pd.DataFrame:

    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)

    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "ansi"]
    last_error = None

    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(
        "Unable to detect file encoding. Please save the file as CSV UTF-8. "
        f"Original error: {last_error}"
    )

def get_variable_groups(df: pd.DataFrame) -> tuple[str, list[str], list[str], list[str], str]:

    if df.shape[1] < 22:
        raise ValueError(
            f"Insufficient number of columns: {df.shape[1]} found, at least 22 required."
        )

    id_col = df.columns[0]
    categorical_cols = df.columns[1:6].tolist()
    continuous_cols = df.columns[6:15].tolist()
    ordinal_cols = df.columns[15:21].tolist()
    event_col_by_position = df.columns[21]

    if EVENT_COL in df.columns:
        event_col = EVENT_COL
    else:
        event_col = event_col_by_position
        warnings.warn(
            f"Column '{EVENT_COL}' not found. Using column 22 '{event_col_by_position}' as event column."
        )

    return id_col, categorical_cols, continuous_cols, ordinal_cols, event_col

def validate_ordinal_columns(ordinal_cols: list[str]) -> None:
    expected_cols = list(ORDINAL_RANGES.keys())

    missing_expected = [col for col in expected_cols if col not in ordinal_cols]
    if missing_expected:
        warnings.warn(
            "The following expected ordinal variables are not in columns 16-21: "
            f"{missing_expected}"
        )

    extra_cols = [col for col in ordinal_cols if col not in ORDINAL_RANGES]
    if extra_cols:
        warnings.warn(
            "The following ordinal columns have no predefined ranges and will only be rounded: "
            f"{extra_cols}"
        )

def prepare_imputation_data(
    df: pd.DataFrame,
    categorical_cols: list[str],
    continuous_cols: list[str],
    ordinal_cols: list[str],
) -> pd.DataFrame:
    work_cols = categorical_cols + continuous_cols + ordinal_cols
    work = df[work_cols].copy()

    for col in categorical_cols:
        work[col] = work[col].astype("category")

    for col in continuous_cols + ordinal_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    object_cols = work.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_cols:
        work[col] = work[col].astype("category")

    return work

def postprocess_imputed_data(
    imputed_work: pd.DataFrame,
    categorical_cols: list[str],
    continuous_cols: list[str],
    ordinal_cols: list[str],
) -> pd.DataFrame:
    imputed_work = imputed_work.copy()

    for col in categorical_cols:
        if isinstance(imputed_work[col].dtype, pd.CategoricalDtype):
            imputed_work[col] = imputed_work[col].astype(str)
            imputed_work.loc[imputed_work[col] == "nan", col] = pd.NA

    for col in continuous_cols:
        imputed_work[col] = pd.to_numeric(imputed_work[col], errors="coerce")

    for col in ordinal_cols:
        imputed_work[col] = pd.to_numeric(imputed_work[col], errors="coerce").round()

        if col in ORDINAL_RANGES:
            lower, upper = ORDINAL_RANGES[col]
            imputed_work[col] = imputed_work[col].clip(lower=lower, upper=upper)

        imputed_work[col] = imputed_work[col].astype("Int64")

    return imputed_work

def impute_external_cohort(df: pd.DataFrame) -> pd.DataFrame:
    id_col, categorical_cols, continuous_cols, ordinal_cols, event_col = get_variable_groups(df)

    print("\nDetected columns:")
    print("ID column:", id_col)
    print("Categorical columns:", categorical_cols)
    print("Continuous columns:", continuous_cols)
    print("Ordinal columns:", ordinal_cols)
    print("Event column:", event_col)

    validate_ordinal_columns(ordinal_cols)

    work_cols = categorical_cols + continuous_cols + ordinal_cols
    exclude_cols = [id_col, event_col]

    overlap = [col for col in work_cols if col in exclude_cols]
    if overlap:
        raise ValueError(
            f"Imputation variables should not include ID or event columns: {overlap}"
        )

    work = prepare_imputation_data(
        df=df,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        ordinal_cols=ordinal_cols,
    )

    print("\nMissing values before imputation:")
    print(work[work_cols].isna().sum())

    total_missing = work[work_cols].isna().sum().sum()
    if total_missing == 0:
        warnings.warn("No missing values detected in imputation variables. Returning original data.")
        return df.copy()

    kernel = mf.ImputationKernel(
        data=work,
        random_state=IMPUTE_SEED,
        save_all_iterations_data=False,
    )

    kernel.mice(IMPUTE_ITERS)
    imputed_work = kernel.complete_data(dataset=0, inplace=False)

    imputed_work = postprocess_imputed_data(
        imputed_work=imputed_work,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        ordinal_cols=ordinal_cols,
    )

    imputed_df = df.copy()
    imputed_df[work_cols] = imputed_work[work_cols]

    imputed_df[id_col] = df[id_col]
    imputed_df[event_col] = df[event_col]

    imputed_df = imputed_df[df.columns]

    print("\nMissing values after imputation:")
    print(imputed_df[work_cols].isna().sum())

    remaining_missing = imputed_df[work_cols].isna().sum().sum()
    if remaining_missing > 0:
        warnings.warn(
            f"{remaining_missing} missing values remain after imputation. "
            "Please check variable types, missingness rate, or original encoding."
        )

    return imputed_df

def main() -> None:
    df = load_data(INPUT_PATH)

    print("\nOriginal data shape:", df.shape)
    print("\nMissing values in original data:")
    print(df.isna().sum())

    imputed_df = impute_external_cohort(df)

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imputed_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nImputation completed.")
    print("Output data shape:", imputed_df.shape)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()