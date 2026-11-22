import pandas as pd
import numpy as np


def read_dataset(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Basic information
    print("DataFrame shape:", df.shape)
    print("Number of patients:", len(df))

    # Column names
    df.columns = ( df.columns.str.replace("2 weeks::", "", regex=False).str.replace("6 months later::", "", regex=False))
    print("\nColumns:")
    print(df.columns.tolist())
    
    # First 5 patients
    print("\nFirst 5 patients:")
    print(df.head())

    # Data types and missing values
    print("\nDataFrame information:")
    df.info()

    print("\nMissing values:")
    print(df.isna().sum())

    target = "ASIAImpairmentScale"
    # Frequency of each class
    class_frequency = df[target].value_counts().sort_index()
    print(class_frequency)
    
    return df
    
def prep_dataset(df):  
    # Convert ASIA Impairment Scale to binary labels
    # Classes 1, 2, 3 -> 1
    # Class 4         -> 0
    df["ASIAImpairmentScale"] = df["ASIAImpairmentScale"].map({
        1: 1,
        2: 1,
        3: 1,
        4: 0
    })

    # X: columns from the beginning through "Biceps brachii_R AL_S4_5"
    start_col ="Biceps brachii_R" 
    end_col = "AL_S4_5"

    X = df.loc[:, start_col:end_col].copy()

    # y: binary ASIA Impairment Scale
    y = df["ASIAImpairmentScale"].copy()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nClass frequency:")
    print(y.value_counts())

    return X, y




import numpy as np
import pandas as pd


def select_rl_predictors(X, method="both"):
    """
    Select/combine right-left paired predictors while preserving
    all unpaired predictors.

    Recognized R/L pairs
    --------------------
    1. Suffix type:
       Biceps brachii_R <-> Biceps brachii_L

    2. TR/TL type:
       TR_C5 <-> TL_C5

    3. AR/AL type:
       AR_C5 <-> AL_C5

    Parameters
    ----------
    X : pandas.DataFrame
        Original predictor DataFrame.

    method : str
        "both"
            Keep both R and L. Returns all original predictors.

        "left"
            For each pair, retain the L-side predictor.

        "right"
            For each pair, retain the R-side predictor.

        "mean"
            Replace each R/L pair by its patient-wise mean.

        "smaller"
            Replace each pair by the smaller patient-wise value.

        "larger"
            Replace each pair by the larger patient-wise value.

    Returns
    -------
    X_selected : pandas.DataFrame
    """

    valid_methods = {
        "both",
        "left",
        "right",
        "mean",
        "smaller",
        "larger"
    }

    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {sorted(valid_methods)}"
        )

    # =========================================================
    # Method 1: BOTH
    # Keep every original predictor.
    # =========================================================
    if method == "both":

        X_selected = X.copy()

        print("Selection method: BOTH")
        print("Original number of predictors:", X.shape[1])
        print("Number of selected predictors:", X_selected.shape[1])

        return X_selected


    # =========================================================
    # Find R/L pairs
    # =========================================================

    pairs = []
    used_columns = set()


    # ---------------------------------------------------------
    # A. Columns ending in _R / _L
    # ---------------------------------------------------------
    for col in X.columns:

        if col.endswith("_R"):

            base = col[:-2]
            left_col = base + "_L"

            if left_col in X.columns:

                pairs.append({
                    "base": base,
                    "R": col,
                    "L": left_col,
                    "type": "suffix"
                })

                used_columns.add(col)
                used_columns.add(left_col)


    # ---------------------------------------------------------
    # B. TR_xxx / TL_xxx
    # ---------------------------------------------------------
    for col in X.columns:

        if col.startswith("TR_"):

            suffix = col[3:]
            left_col = "TL_" + suffix

            if left_col in X.columns:

                pairs.append({
                    "base": suffix,
                    "R": col,
                    "L": left_col,
                    "type": "T"
                })

                used_columns.add(col)
                used_columns.add(left_col)


    # ---------------------------------------------------------
    # C. AR_xxx / AL_xxx
    # ---------------------------------------------------------
    for col in X.columns:

        if col.startswith("AR_"):

            suffix = col[3:]
            left_col = "AL_" + suffix

            if left_col in X.columns:

                pairs.append({
                    "base": suffix,
                    "R": col,
                    "L": left_col,
                    "type": "A"
                })

                used_columns.add(col)
                used_columns.add(left_col)


    # =========================================================
    # Identify unpaired predictors
    # =========================================================

    unpaired_columns = [
        col for col in X.columns
        if col not in used_columns
    ]


    print("\nNumber of R/L pairs:", len(pairs))
    print("Paired predictors:", len(used_columns))
    print("Unpaired predictors:", len(unpaired_columns))


    # =========================================================
    # Build new DataFrame while preserving original ordering
    # =========================================================

    output = {}
    processed = set()


    for col in X.columns:

        # -----------------------------------------------------
        # Unpaired predictor
        # -----------------------------------------------------
        if col not in used_columns:

            output[col] = X[col]
            continue


        # Already handled as part of a pair
        if col in processed:
            continue


        # Find the pair containing this column
        pair = next(
            p for p in pairs
            if col == p["R"] or col == p["L"]
        )

        R_col = pair["R"]
        L_col = pair["L"]

        R = X[R_col]
        L = X[L_col]

        processed.add(R_col)
        processed.add(L_col)


        # =====================================================
        # LEFT
        # =====================================================

        if method == "left":

            output[L_col] = L


        # =====================================================
        # RIGHT
        # =====================================================

        elif method == "right":

            output[R_col] = R


        # =====================================================
        # MEAN
        # =====================================================

        elif method == "mean":

            if pair["type"] == "suffix":
                new_name = pair["base"] + "_mean"

            elif pair["type"] == "T":
                new_name = "T_" + pair["base"] + "_mean"

            else:
                new_name = "A_" + pair["base"] + "_mean"

            output[new_name] = pd.concat(
                [R, L],
                axis=1
            ).mean(axis=1)


        # =====================================================
        # SMALLER
        # =====================================================

        elif method == "smaller":

            if pair["type"] == "suffix":
                new_name = pair["base"] + "_smaller"

            elif pair["type"] == "T":
                new_name = "T_" + pair["base"] + "_smaller"

            else:
                new_name = "A_" + pair["base"] + "_smaller"

            output[new_name] = pd.concat(
                [R, L],
                axis=1
            ).min(axis=1)


        # =====================================================
        # LARGER
        # =====================================================

        elif method == "larger":

            if pair["type"] == "suffix":
                new_name = pair["base"] + "_larger"

            elif pair["type"] == "T":
                new_name = "T_" + pair["base"] + "_larger"

            else:
                new_name = "A_" + pair["base"] + "_larger"

            output[new_name] = pd.concat(
                [R, L],
                axis=1
            ).max(axis=1)


    X_selected = pd.DataFrame(
        output,
        index=X.index
    )


    print("\nSelection method:", method.upper())

    print(
        "Original number of predictors:",
        X.shape[1]
    )

    print(
        "Number of selected predictors:",
        X_selected.shape[1]
    )


    return X_selected
    
    
if __name__=="__main__":
    # Path to the Excel file
    file_path = "data_SCI_eng.xlsx"
    df = read_dataset(file_path) 
    X, y = prep_dataset(df)

    X_both = select_rl_predictors(X, method="both")
    X_left = select_rl_predictors(X, method="left")
    X_right = select_rl_predictors(X, method="right")
    X_mean = select_rl_predictors(X, method="mean")
    X_smaller = select_rl_predictors(X, method="smaller")
    X_larger = select_rl_predictors(X, method="larger") 
         
    selected_predictor={"both": X_both, "left": X_left, "right": X_right, "mean": X_mean, "smaller": X_smaller, "larger": X_larger }
    
    #print(selected_predictor["both"])

    for key, value in selected_predictor.items():
        print(key)
        print(value)
    
    
    
