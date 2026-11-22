import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from scipy.special import logit
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

from preselect_predictor import select_rl_predictors


def preselect_motor_and_sensory(X, method):
    """Apply the original motor preselector and bilateral sensory selection.

    Sensory columns are matched as TR_/TL_ (T measurement) and AR_/AL_
    (A measurement), pairing columns with identical spinal-segment suffixes.
    The existing select_rl_predictors is used unchanged for motor predictors.
    """
    allowed = {"both", "left", "right", "mean", "smaller", "larger"}
    if method not in allowed:
        raise ValueError(f"Unknown preselection method: {method!r}")
    if X.columns.duplicated().any():
        raise ValueError("Duplicate predictor names are not supported")
    if method == "both":
        return X.copy()

    sensory_pattern = re.compile(r"^(TR|TL|AR|AL)_(.+)$")
    sensory_columns = [c for c in X.columns if sensory_pattern.fullmatch(str(c))]
    motor_columns = [c for c in X.columns if c not in sensory_columns]

    # Do not change the original motor preselection implementation.
    motor_selected = select_rl_predictors(X.loc[:, motor_columns].copy(), method=method)
    if not isinstance(motor_selected, pd.DataFrame):
        raise TypeError("Motor preselector must return a pandas DataFrame")
    if not motor_selected.index.equals(X.index):
        raise ValueError("Motor preselector changed the patient index")

    # Each sensory measurement (T or A) is paired by spinal segment.
    groups = {}
    for col in sensory_columns:
        prefix, segment = sensory_pattern.fullmatch(str(col)).groups()
        kind = prefix[0]
        side = prefix[1]
        groups.setdefault((kind, segment), {})[side] = col

    sensory_values = {}
    positions = {col: i for i, col in enumerate(X.columns)}
    sensory_positions = {}
    for (kind, segment), sides in groups.items():
        if set(sides) != {"R", "L"}:
            raise ValueError(
                f"Incomplete sensory left/right pair for {kind}_{segment}: "
                f"found {sorted(sides)}"
            )
        right = X[sides["R"]]
        left = X[sides["L"]]
        if method == "right":
            name, values = sides["R"], right
        elif method == "left":
            name, values = sides["L"], left
        else:
            # Merge the right/left observations into a single predictor.
            name = f"{kind}_{segment}"
            pair = pd.concat([right, left], axis=1)
            if method == "mean":
                values = pair.mean(axis=1)
            elif method == "smaller":
                values = pair.min(axis=1)
            else:  # larger
                values = pair.max(axis=1)
        if name in sensory_values or name in motor_selected.columns:
            raise ValueError(f"Duplicate preselected predictor name: {name}")
        sensory_values[name] = values
        sensory_positions[name] = min(positions[sides["R"]], positions[sides["L"]])

    sensory_selected = pd.DataFrame(sensory_values, index=X.index)
    combined = pd.concat([motor_selected, sensory_selected], axis=1)
    # Retain original input ordering where the motor helper preserves names.
    original_positions = dict(positions)
    for col in motor_selected.columns:
        if col not in original_positions:
            # Aggregated motor columns occupy the position of their R/L pair.
            possible = [f"{col}_R", f"{col}_L"]
            original_positions[col] = min(
                (positions[p] for p in possible if p in positions),
                default=len(positions),
            )
    original_positions.update(sensory_positions)
    ordered_columns = sorted(combined.columns, key=lambda col: original_positions.get(col, len(positions)))
    return combined.loc[:, ordered_columns]


class DomainCorrelationReducer(BaseEstimator, TransformerMixin):
    """
    Reduce redundant predictors using TRAINING DATA ONLY.

    Stage 1: Domain-based grouping
        Variables beginning with TR_, TL_, AR_, or AL_ and sharing the same
        suffix are treated as one biological/measurement group.
        Example:
            TR_C5, TL_C5, AR_C5, AL_C5
        -> keep one representative.

    Stage 2: Generic correlation filtering
        Among the remaining predictors, variables connected by absolute
        Pearson correlation >= corr_threshold are treated as one correlated
        component, and one representative is retained.

    Representative rule (unsupervised):
        1. lowest missing-value fraction
        2. highest variance
        3. earliest original column position as a deterministic tie-breaker

    No outcome y is used for either reduction stage.
    """

    def __init__(
        self,
        corr_threshold=0.85,
        domain_prefixes=("TR", "TL", "AR", "AL"),
        verbose=False,
    ):
        self.corr_threshold = corr_threshold
        self.domain_prefixes = domain_prefixes
        self.verbose = verbose

    @staticmethod
    def _choose_representative(X, columns, original_order):
        """Choose one representative without using y."""
        ranking = []

        for col in columns:
            missing_fraction = X[col].isna().mean()

            # pandas variance safely ignores NaN.
            variance = X[col].var()
            if pd.isna(variance):
                variance = -np.inf

            ranking.append(
                (
                    missing_fraction,     # smaller is better
                    -variance,            # larger variance is better
                    original_order[col],  # earlier position breaks ties
                    col,
                )
            )

        ranking.sort()
        return ranking[0][3]

    def _domain_key(self, column):
        """
        Return a grouping key for TR/TL/AR/AL variables.

        Examples:
            TR_C5 -> C5
            TL_C5 -> C5
            AR_C5 -> C5
            AL_C5 -> C5

        Columns not starting with one of these prefixes return None.
        """
        escaped = "|".join(re.escape(p) for p in self.domain_prefixes)
        match = re.match(rf"^(?:{escaped})_(.+)$", str(column))
        return match.group(1) if match else None

    @staticmethod
    def _connected_components(corr, threshold):
        """Connected components of the high-correlation graph."""
        columns = list(corr.columns)
        adjacency = {col: set() for col in columns}

        for i, col_i in enumerate(columns):
            for j in range(i + 1, len(columns)):
                col_j = columns[j]
                r = corr.iloc[i, j]

                if pd.notna(r) and r >= threshold:
                    adjacency[col_i].add(col_j)
                    adjacency[col_j].add(col_i)

        components = []
        visited = set()

        for start in columns:
            if start in visited:
                continue

            stack = [start]
            component = []
            visited.add(start)

            while stack:
                node = stack.pop()
                component.append(node)

                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            components.append(component)

        return components

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "DomainCorrelationReducer expects a pandas DataFrame as input."
            )

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        original_order = {col: i for i, col in enumerate(X.columns)}

        # ------------------------------------------------------------
        # Stage 1: explicitly group TR/TL/AR/AL variables by suffix.
        # ------------------------------------------------------------
        domain_groups = {}
        non_domain_columns = []

        for col in X.columns:
            key = self._domain_key(col)

            if key is None:
                non_domain_columns.append(col)
            else:
                domain_groups.setdefault(key, []).append(col)

        domain_selected = []
        self.domain_groups_ = {}
        self.domain_representatives_ = {}
        self.domain_removed_ = {}

        for key, columns in domain_groups.items():
            representative = self._choose_representative(
                X, columns, original_order
            )

            domain_selected.append(representative)
            self.domain_groups_[key] = list(columns)
            self.domain_representatives_[key] = representative

            for col in columns:
                if col != representative:
                    self.domain_removed_[col] = representative

        stage1_columns = [
            col
            for col in X.columns
            if col in non_domain_columns or col in domain_selected
        ]

        # ------------------------------------------------------------
        # Stage 2: generic high-correlation reduction.
        # ------------------------------------------------------------
        X_stage1 = X[stage1_columns]
        corr = X_stage1.corr(method="pearson").abs()

        components = self._connected_components(
            corr,
            self.corr_threshold,
        )

        self.correlation_groups_ = []
        self.correlation_removed_ = {}
        final_selected = []

        for component in components:
            representative = self._choose_representative(
                X_stage1,
                component,
                original_order,
            )

            final_selected.append(representative)

            if len(component) > 1:
                self.correlation_groups_.append(
                    {
                        "representative": representative,
                        "members": list(component),
                    }
                )

                for col in component:
                    if col != representative:
                        self.correlation_removed_[col] = representative

        # Preserve the original spreadsheet order.
        selected_set = set(final_selected)
        self.selected_features_ = [
            col for col in X.columns if col in selected_set
        ]

        if len(self.selected_features_) < 4:
            raise ValueError(
                "Correlation/group filtering retained fewer than 4 predictors. "
                "Lower corr_threshold or review the domain grouping rule."
            )

        if self.verbose:
            print(
                f"Domain/correlation reducer: "
                f"{X.shape[1]} -> {len(self.selected_features_)} predictors"
            )

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "DomainCorrelationReducer expects a pandas DataFrame as input."
            )

        missing = [
            col for col in self.selected_features_
            if col not in X.columns
        ]
        if missing:
            raise KeyError(
                "Columns learned during fit are missing during transform: "
                f"{missing}"
            )

        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


def read_dataset(file_path):
    """Read the SCI spreadsheet and simplify the time-point prefixes."""
    df = pd.read_excel(file_path)

    print("DataFrame shape:", df.shape)
    print("Number of patients:", len(df))

    # Remove leading/trailing spaces first, then remove the two prefixes.
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("2 weeks::", "", regex=False)
        .str.replace("6 months later::", "", regex=False)
    )

    # Fail early if renaming unexpectedly creates duplicates.
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "Duplicate column names were created after removing time-point prefixes: "
            f"{duplicated}"
        )

    target = "ASIAImpairmentScale"
    print("\nOriginal ASIA class frequency:")
    print(df[target].value_counts(dropna=False).sort_index())

    return df


def prep_dataset(df):
    """Create X and the binary y outcome."""
    target = "ASIAImpairmentScale"

    # Classes 1, 2, 3 -> 0; class 4 -> 1.
    y = df[target].map({1: 0, 2: 0, 3: 0, 4: 1})

    # Predictors: all consecutive variables from Biceps brachii_R to AL_S4_5.
    start_col = "Biceps brachii_R"
    end_col = "AL_S4_5"

    if start_col not in df.columns:
        raise KeyError(f'Column "{start_col}" was not found.')
    if end_col not in df.columns:
        raise KeyError(f'Column "{end_col}" was not found.')

    # Use integer positions so slicing is robust even if labels are non-monotonic.
    start_idx = df.columns.get_loc(start_col)
    end_idx = df.columns.get_loc(end_col)

    if not isinstance(start_idx, (int, np.integer)) or not isinstance(
        end_idx, (int, np.integer)
    ):
        raise ValueError("start_col or end_col is duplicated in the DataFrame.")

    if start_idx > end_idx:
        raise ValueError("start_col occurs after end_col in the spreadsheet.")

    X = df.iloc[:, start_idx : end_idx + 1].copy()

    # Keep only patients whose outcome belongs to classes 1-4.
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    # Logistic regression + median imputation require numeric predictors.
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(
            "Non-numeric predictors found in X. Encode or remove these columns first: "
            f"{non_numeric}"
        )

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)
    print("\nBinary class frequency:")
    print(y.value_counts().sort_index())

    return X, y


def prefer_simpler_refit(cv_results, tolerance=0.01):
    """Select highest-AUC inner-CV configuration (all use three predictors)."""
    scores = np.asarray(cv_results["mean_test_score"], dtype=float)
    n_features = np.asarray(cv_results["param_selector__max_features"], dtype=int)

    best_score = np.nanmax(scores)
    eligible = np.flatnonzero(scores >= best_score - tolerance)

    # Among near-best models, prefer the smallest number of predictors.
    min_features = np.min(n_features[eligible])
    eligible = eligible[n_features[eligible] == min_features]

    # Within that feature count, choose the highest AUC.
    return int(eligible[np.nanargmax(scores[eligible])])



def bootstrap_oof_ci(
    y_true,
    y_pred,
    y_prob,
    n_bootstrap=5000,
    confidence_level=0.95,
    random_state=42,
):
    """
    Patient-level percentile bootstrap confidence intervals from pooled
    out-of-fold predictions.

    Each bootstrap replicate samples patients with replacement from the pooled
    outer-CV predictions. Replicates containing only one true class are skipped
    because ROC-AUC is undefined for them.

    Returns
    -------
    ci_df : pandas.DataFrame
        Point estimate and bootstrap percentile CI for each metric.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if not (len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError("y_true, y_pred, and y_prob must have the same length.")
    if len(y_true) == 0:
        raise ValueError("No out-of-fold predictions were supplied.")
    if np.unique(y_true).size < 2:
        raise ValueError("Both outcome classes are required for ROC-AUC.")

    def calculate_metrics(yt, yp, ypr):
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        return {
            "Accuracy": accuracy_score(yt, yp),
            "AUC": roc_auc_score(yt, ypr),
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision_score(yt, yp, zero_division=0),
            "F1": f1_score(yt, yp, zero_division=0),
            "MCC": matthews_corrcoef(yt, yp),
        }

    point_estimates = calculate_metrics(y_true, y_pred, y_prob)
    bootstrap_values = {metric: [] for metric in point_estimates}

    rng = np.random.default_rng(random_state)
    n_patients = len(y_true)

    valid_replicates = 0
    attempted_replicates = 0

    while valid_replicates < n_bootstrap:
        attempted_replicates += 1
        indices = rng.integers(0, n_patients, size=n_patients)

        yt = y_true[indices]

        # AUC is undefined when a bootstrap sample contains only one class.
        if np.unique(yt).size < 2:
            continue

        yp = y_pred[indices]
        ypr = y_prob[indices]
        metrics = calculate_metrics(yt, yp, ypr)

        for metric, value in metrics.items():
            if np.isfinite(value):
                bootstrap_values[metric].append(value)

        valid_replicates += 1

    alpha = 1.0 - confidence_level
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    rows = []
    for metric, point in point_estimates.items():
        values = np.asarray(bootstrap_values[metric], dtype=float)

        rows.append(
            {
                "Metric": metric,
                "Estimate": point,
                "CI95_lower": np.percentile(values, lower_q),
                "CI95_upper": np.percentile(values, upper_q),
                "Bootstrap_samples": len(values),
            }
        )

    ci_df = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("PATIENT-LEVEL BOOTSTRAP 95% CI FROM POOLED OOF PREDICTIONS")
    print("=" * 72)
    print(f"Patients             : {n_patients}")
    print(f"Valid bootstrap runs : {valid_replicates}")
    print(f"Attempts             : {attempted_replicates}")
    print(ci_df.round(3).to_string(index=False))

    return ci_df

def evaluate_nested_cv(
    X, y, n_splits=3, random_state=42, out_dir=None, bootstrap_samples=5000
):
    """
    Nested 3-fold CV.

    Inner CV:
      - domain-based reduction: one representative from each TR/TL/AR/AL group
      - generic correlation filtering among remaining predictors
      - median imputation
      - standardization
      - L1 logistic-regression ranking
      - EXACTLY 3 predictors via SelectFromModel(max_features=3)
      - tune selector C and final logistic-regression C
      - optimize the selector and classifier C values by ROC-AUC

    Outer CV:
      - unbiased evaluation on held-out patients

    IMPORTANT:
      Domain grouping and correlation filtering are inside the sklearn Pipeline,
      so they are refit independently in every inner- and outer-training split.
    """
    selector_c_grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    classifier_c_grid = [0.1, 0.3, 1.0, 3.0, 10.0]

    # Generic correlation threshold applied AFTER the explicit
    # TR/TL/AR/AL grouping. Adjust to 0.90 for a less aggressive filter.
    corr_threshold = 0.85

    outer_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    results = []
    selected_features_each_fold = []
    selected_coefficients_each_fold = []

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        print("\n" + "=" * 72)
        print(f"OUTER FOLD {fold}")
        print("=" * 72)

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        selector = SelectFromModel(
            estimator=LogisticRegression(
                l1_ratio=1.0,
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            ),
            # -inf means max_features controls the exact number retained.
            threshold=-np.inf,
            max_features=3,
        )

        reducer = DomainCorrelationReducer(
            corr_threshold=corr_threshold,
            domain_prefixes=("TR", "TL", "AR", "AL"),
            verbose=False,
        )

        pipe = Pipeline(
            steps=[
                ("reducer", reducer),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("selector", selector),
                ("classifier", LogisticRegression(
                    l1_ratio=0.0,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                )),
            ]
        )

        param_grid = {
            "selector__max_features": [3],
            "selector__estimator__C": selector_c_grid,
            "classifier__C": classifier_c_grid,
        }

        inner_cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + fold,
        )

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=lambda cv_results: prefer_simpler_refit(
                cv_results, tolerance=0.01
            ),
            return_train_score=False,
        )

        # IMPORTANT: all preprocessing and feature selection are fit only on
        # this outer fold's training patients.
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        reduced_features = best_model.named_steps["reducer"].get_feature_names_out()
        selected_mask = best_model.named_steps["selector"].get_support()
        selected_features = reduced_features[selected_mask].tolist()

        if len(selected_features) != 3:
            raise RuntimeError(
                f"Expected exactly 3 selected features, got {len(selected_features)}."
            )

        final_coefs = best_model.named_steps["classifier"].coef_[0]
        selected_coefficients = dict(zip(selected_features, final_coefs))

        selected_features_each_fold.append(selected_features)
        selected_coefficients_each_fold.append(selected_coefficients)

        reducer_fit = best_model.named_steps["reducer"]

        print(
            "Predictors after domain/correlation reduction:",
            len(reducer_fit.selected_features_),
        )
        print("Selected number of predictors:", len(selected_features))
        print("Best selector C:", grid.best_params_["selector__estimator__C"])
        print("Best classifier C:", grid.best_params_["classifier__C"])

        print("\nTR/TL/AR/AL representatives in this outer fold:")
        if reducer_fit.domain_representatives_:
            for key, representative in reducer_fit.domain_representatives_.items():
                members = reducer_fit.domain_groups_[key]
                if len(members) > 1:
                    print(
                        f"  {key:25s} -> {representative:25s} "
                        f"from {members}"
                    )
        else:
            print("  No TR/TL/AR/AL groups found.")

        print("\nAdditional correlated predictors removed in this outer fold:")
        if reducer_fit.correlation_removed_:
            for removed, representative in reducer_fit.correlation_removed_.items():
                print(f"  {removed:40s} -> {representative}")
        else:
            print("  None.")
        print("\nSelected predictors and final-model coefficients:")
        for feature, coef in selected_coefficients.items():
            print(f"  {feature:45s} {coef: .4f}")

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        sensitivity = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        results.append(
            {
                "Fold": fold,
                "N_features": len(selected_features),
                "Accuracy": accuracy,
                "AUC": auc,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Precision": precision,
                "F1": f1,
                "MCC": mcc,
            }
        )

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_prob_all.extend(y_prob.tolist())

        print("\nFold performance:")
        print(f"  Accuracy    : {accuracy:.3f}")
        print(f"  AUC         : {auc:.3f}")
        print(f"  Sensitivity : {sensitivity:.3f}")
        print(f"  Specificity : {specificity:.3f}")
        print(f"  Precision   : {precision:.3f}")
        print(f"  F1          : {f1:.3f}")
        print(f"  MCC         : {mcc:.3f}")

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 72)
    print("3-FOLD NESTED CROSS-VALIDATION RESULTS")
    print("=" * 72)
    print(results_df.to_string(index=False))

    metric_cols = [
        "Accuracy",
        "AUC",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
        "MCC",
    ]

    print("\nMean performance:")
    print(results_df[metric_cols].mean().round(3))

    print("\nStandard deviation:")
    print(results_df[metric_cols].std().round(3))

    print("\nOverall out-of-fold performance:")
    print(f"  Accuracy : {accuracy_score(y_true_all, y_pred_all):.3f}")
    print(f"  AUC      : {roc_auc_score(y_true_all, y_prob_all):.3f}")
    print(f"  MCC      : {matthews_corrcoef(y_true_all, y_pred_all):.3f}")

    # Save pooled out-of-fold predictions and calculate patient-level
    # bootstrap confidence intervals.
    oof_df = pd.DataFrame({
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_prob": y_prob_all,
    })

    bootstrap_ci_df = bootstrap_oof_ci(
        y_true_all,
        y_pred_all,
        y_prob_all,
        n_bootstrap=bootstrap_samples,
        confidence_level=0.95,
        random_state=random_state,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
        bootstrap_ci_df.to_csv(out_dir / "bootstrap_95CI.csv", index=False)

    # Feature stability across the three independently selected outer folds.
    all_selected = [
        feature
        for fold_features in selected_features_each_fold
        for feature in fold_features
    ]

    selection_frequency = (
        pd.Series(all_selected, dtype="object")
        .value_counts()
        .rename_axis("Feature")
        .reset_index(name="Selected_folds")
    )
    selection_frequency["Selection_percentage"] = (
        selection_frequency["Selected_folds"] / n_splits * 100
    ).round(1)

    print("\n" + "=" * 72)
    print("FEATURE-SELECTION STABILITY")
    print("=" * 72)
    print(selection_frequency.to_string(index=False))

    stable_features = selection_frequency.loc[
        selection_frequency["Selected_folds"] >= 2, "Feature"
    ].tolist()

    print("\nStable predictors (selected in at least 2 of 3 outer folds):")
    if stable_features:
        for feature in stable_features:
            print(" ", feature)
    else:
        print("  None. Feature selection is unstable across folds.")

    return (
        results_df,
        selection_frequency,
        selected_features_each_fold,
        bootstrap_ci_df,
        oof_df,
    )


METRIC_COLUMNS = [
    "Accuracy", "AUC", "Sensitivity", "Specificity", "Precision", "F1", "MCC"
]


def summarize_repeated_metrics(repeat_metrics, n_bootstrap=10000, random_state=42,
                               metric_columns=None):
    """Bootstrap 95% CIs for the mean over independent CV repetitions.

    Resampling unit is the entire nested-CV repetition, NOT an outer fold or
    an OOF patient prediction. This captures variability from CV partitions.
    """
    if n_bootstrap < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    rng = np.random.default_rng(random_state)
    rows = []
    for metric in (METRIC_COLUMNS if metric_columns is None else metric_columns):
        values = pd.to_numeric(repeat_metrics[metric], errors="coerce").to_numpy()
        values = values[np.isfinite(values)]
        n = len(values)
        if not n:
            rows.append({"Metric": metric, "Mean": np.nan, "Std": np.nan,
                         "CI95_lower": np.nan, "CI95_upper": np.nan,
                         "N_repeats": 0, "Bootstrap_samples": 0})
            continue
        mean = float(values.mean())
        if n < 2:
            # A one-run analysis cannot estimate between-repeat uncertainty.
            std, lower, upper, valid_samples = np.nan, np.nan, np.nan, 0
        else:
            std = float(values.std(ddof=1))
            # Chunk to avoid large allocations when n_time is large.
            boot_means = np.empty(n_bootstrap)
            for start in range(0, n_bootstrap, 1000):
                stop = min(start + 1000, n_bootstrap)
                indices = rng.integers(0, n, size=(stop - start, n))
                boot_means[start:stop] = values[indices].mean(axis=1)
            lower, upper = np.percentile(boot_means, [2.5, 97.5])
            valid_samples = n_bootstrap
        rows.append({"Metric": metric, "Mean": mean, "Std": std,
                     "CI95_lower": lower, "CI95_upper": upper,
                     "N_repeats": n, "Bootstrap_samples": valid_samples})
    return pd.DataFrame(rows)



CALIBRATION_METRIC_COLUMNS = ["Brier_score", "Calibration_intercept", "Calibration_slope"]


def calculate_calibration_metrics(y_true, y_prob):
    """Compute Brier score, calibration-in-the-large intercept, and slope.

    The intercept is fitted with the original log-odds as an OFFSET (slope=1).
    The slope is fitted jointly with a free intercept. The two intercepts are
    therefore not the same parameter. Return NaN for non-estimable coefficients
    (e.g. complete separation), rather than reporting misleading extremes.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("y_true and y_prob must be nonempty 1-D arrays of equal length")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError("Calibration requires binary outcomes and finite probabilities in [0, 1]")
    result = {"Brier_score": float(brier_score_loss(y, p)),
              "Calibration_intercept": np.nan, "Calibration_slope": np.nan}
    if np.unique(y).size < 2:
        return result
    # Clipping is necessary only for logit regression, not for the Brier score.
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PerfectSeparationWarning)
        try:
            fit = sm.GLM(y, np.ones((len(y), 1)),
                         family=sm.families.Binomial(), offset=lp).fit()
            if fit.converged and np.isfinite(fit.params[0]):
                result["Calibration_intercept"] = float(fit.params[0])
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            pass
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PerfectSeparationWarning)
        try:
            fit = sm.GLM(y, sm.add_constant(lp, has_constant="add"),
                         family=sm.families.Binomial()).fit()
            separated = any(issubclass(w.category, PerfectSeparationWarning) for w in caught)
            if fit.converged and not separated and np.all(np.isfinite(fit.params)):
                result["Calibration_slope"] = float(fit.params[1])
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            pass
    return result


def plot_repeated_calibration(all_oof_df, out_dir, n_bins=10,
                              n_bootstrap=10000, random_state=42):
    """Reliability diagram using held-out predictions from repeated nested CV.

    Bin edges are fixed across repetitions, including empty-bin handling.
    Bootstrap units are whole CV repetitions, never individual pooled rows;
    since repetitions reuse the same patients the ribbon describes variation
    of the mean curve across random CV partitions, NOT external validation CI.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    required = {"Repeat", "y_true", "y_prob"}
    if not required.issubset(all_oof_df.columns):
        raise ValueError(f"OOF data missing columns: {sorted(required - set(all_oof_df))}")
    data = all_oof_df.copy()
    if not np.all(np.isfinite(data["y_prob"])) or not data["y_prob"].between(0, 1).all():
        raise ValueError("Predicted probabilities must be finite and between 0 and 1")
    if not data["y_true"].isin([0, 1]).all():
        raise ValueError("Calibration requires binary 0/1 outcomes")

    edges = np.linspace(0, 1, n_bins + 1)
    data["Bin"] = np.minimum(np.searchsorted(edges, data["y_prob"], side="right") - 1,
                             n_bins - 1)
    # For values exactly 0 the searchsorted result is bin 0.
    run_rows = []
    for rep, group in data.groupby("Repeat", sort=True):
        seed = group["Random_state"].iloc[0] if "Random_state" in group else np.nan
        for bin_idx in range(n_bins):
            g = group[group["Bin"] == bin_idx]
            run_rows.append({
                "Repeat": rep, "Random_state": seed, "Bin": bin_idx + 1,
                "Bin_lower": edges[bin_idx], "Bin_upper": edges[bin_idx + 1],
                "N": len(g),
                "Mean_predicted_probability": g["y_prob"].mean() if len(g) else np.nan,
                "Observed_positive_fraction": g["y_true"].mean() if len(g) else np.nan,
            })
    bins_df = pd.DataFrame(run_rows)
    bins_df.to_csv(out_dir / "calibration_bins_by_repeat.csv", index=False)

    rng = np.random.default_rng(random_state)
    # Across each bin, equal-weight nonempty repetitions; same fixed bins per run.
    # Whole-repeat resampling retains cross-bin correlation in each draw.
    reps = sorted(data["Repeat"].unique())
    pred = bins_df.pivot(index="Repeat", columns="Bin", values="Mean_predicted_probability").loc[reps].to_numpy()
    obs = bins_df.pivot(index="Repeat", columns="Bin", values="Observed_positive_fraction").loc[reps].to_numpy()
    counts = bins_df.pivot(index="Repeat", columns="Bin", values="N").loc[reps].to_numpy()
    rows = []
    for j in range(n_bins):
        good = np.isfinite(obs[:, j]) & np.isfinite(pred[:, j])
        valid_obs, valid_pred = obs[good, j], pred[good, j]
        if not len(valid_obs):
            lo = hi = x = y = np.nan
        else:
            x, y = float(valid_pred.mean()), float(valid_obs.mean())
            if len(reps) < 2:
                lo = hi = np.nan
            else:
                boot = []
                # Resample full repetitions, then omit empty bins in each sample.
                for _ in range(n_bootstrap):
                    sample_idx = rng.integers(0, len(reps), size=len(reps))
                    yy = obs[sample_idx, j]
                    yy = yy[np.isfinite(yy)]
                    if len(yy):
                        boot.append(yy.mean())
                lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
        rows.append({
            "Bin": j + 1, "Bin_lower": edges[j], "Bin_upper": edges[j + 1],
            "N_repeats_nonempty": int(good.sum()), "Total_predictions": int(counts[:, j].sum()),
            "Mean_predicted_probability": x, "Mean_observed_positive_fraction": y,
            "CI95_lower": lo, "CI95_upper": hi,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "calibration_curve_mean_95CI.csv", index=False)

    # One overall reliability curve from ALL held-out predictions combined.
    # Every patient appears once per repetition, so these rows are dependent;
    # this is a descriptive pooled curve rather than an external-validation CI.
    pooled_rows = []
    for j in range(n_bins):
        g = data.loc[data["Bin"] == j]
        pooled_rows.append({
            "Bin": j + 1,
            "Bin_lower": edges[j],
            "Bin_upper": edges[j + 1],
            "N_predictions": len(g),
            "Mean_predicted_probability": g["y_prob"].mean() if len(g) else np.nan,
            "Observed_positive_fraction": g["y_true"].mean() if len(g) else np.nan,
        })
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out_dir / "calibration_pooled_all_repeats.csv", index=False)

    fig_all, ax_all = plt.subplots(figsize=(7.1, 6.3))
    ax_all.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    nonempty = pooled_df["N_predictions"] > 0
    ax_all.plot(
        pooled_df.loc[nonempty, "Mean_predicted_probability"],
        pooled_df.loc[nonempty, "Observed_positive_fraction"],
        marker="o", color="tab:blue", linewidth=2.3, markersize=6,
        label=f"Pooled OOF calibration ({len(reps)} repetitions)",
    )
    ax_all.set(xlim=(0, 1), ylim=(0, 1),
               xlabel="Mean predicted probability", ylabel="Observed positive fraction",
               title="Overall calibration: all nested-CV repetitions pooled")
    ax_all.grid(alpha=0.2)
    ax_all.legend(loc="best", fontsize=9)
    fig_all.tight_layout()
    fig_all.savefig(out_dir / "calibration_pooled_all_repeats.png", dpi=300)
    fig_all.savefig(out_dir / "calibration_pooled_all_repeats.pdf")
    plt.close(fig_all)
    print(f"Overall pooled calibration figure: {out_dir / 'calibration_pooled_all_repeats.png'}")

    calibration_rows = []
    for rep, group in data.groupby("Repeat", sort=True):
        row = {"Repeat": rep}
        if "Random_state" in group:
            row["Random_state"] = group["Random_state"].iloc[0]
        row.update(calculate_calibration_metrics(group["y_true"], group["y_prob"]))
        calibration_rows.append(row)
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(out_dir / "calibration_metrics_by_repeat.csv", index=False)
    calibration_df[["Repeat", "Brier_score"]].to_csv(
        out_dir / "calibration_brier_by_repeat.csv", index=False)
    calibration_summary = summarize_repeated_metrics(
        calibration_df, n_bootstrap=n_bootstrap, random_state=random_state + 1,
        metric_columns=CALIBRATION_METRIC_COLUMNS)
    calibration_summary.to_csv(out_dir / "calibration_metrics_mean_95CI.csv", index=False)
    print("\nCalibration metrics across nested-CV repetitions:")
    print(calibration_summary.round(4).to_string(index=False))
    if calibration_df[["Calibration_intercept", "Calibration_slope"]].isna().any().any():
        print("WARNING: Some calibration coefficients were non-estimable; "
              "their summaries exclude NaN repetitions.")

    fig, ax = plt.subplots(figsize=(7.1, 6.3))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1.4,
            label="Perfect calibration")
    for rep in reps:
        r = bins_df[bins_df["Repeat"] == rep]
        good = r["N"].to_numpy() > 0
        if good.any():
            ax.plot(r.loc[good, "Mean_predicted_probability"],
                    r.loc[good, "Observed_positive_fraction"],
                    color="steelblue", alpha=0.18, lw=0.8)
    good = summary["N_repeats_nonempty"].to_numpy() > 0
    x = summary.loc[good, "Mean_predicted_probability"].to_numpy()
    y = summary.loc[good, "Mean_observed_positive_fraction"].to_numpy()
    lower = summary.loc[good, "CI95_lower"].to_numpy()
    upper = summary.loc[good, "CI95_upper"].to_numpy()
    if len(reps) > 1:
        ax.fill_between(x, lower, upper, color="darkorange", alpha=0.22,
                        label="95% bootstrap CI of mean curve")
    ax.plot(x, y, marker="o", markersize=5.5, color="darkorange", lw=2.3,
            label=f"Mean calibration ({len(reps)} repetitions)")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Observed positive fraction", title="Repeated nested-CV calibration")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_repeated_nested_cv.png", dpi=300)
    fig.savefig(out_dir / "calibration_repeated_nested_cv.pdf")
    plt.close(fig)
    print(f"Calibration figure: {out_dir / 'calibration_repeated_nested_cv.png'}")
    print(f"Mean Brier score across repetitions: {calibration_df['Brier_score'].mean():.4f}")
    return bins_df, summary, calibration_df


def main():
    parser = argparse.ArgumentParser(
        description="Repeated nested CV for SCI L1/logistic regression."
    )
    parser.add_argument("--preselect", required=True,
                        choices=["both", "left", "right", "mean", "smaller", "larger"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--input", default="data_SCI_eng.xlsx")
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42,
                        help="Starting seed; repeat i uses random_state + i.")
    parser.add_argument("--calibration_bins", type=int, default=10,
                        help="Number of equal-width calibration bins (default: 10).")
    parser.add_argument("--n_time", type=int, required=True,
                        help="Number of independent nested CV repetitions.")
    parser.add_argument("--bootstrap_samples", type=int, default=5000,
                        help="Bootstrap resamples per run (patients) and for aggregate CI (runs).")
    args = parser.parse_args()
    if args.n_time < 1:
        parser.error("--n_time must be >= 1")
    if args.n_splits < 2:
        parser.error("--n_splits must be >= 2")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap_samples must be >= 1")
    if args.calibration_bins < 2:
        parser.error("--calibration_bins must be >= 2")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("SCI L1 LOGISTIC REGRESSION — REPEATED NESTED CROSS-VALIDATION")
    print("=" * 72)
    print(f"Input: {args.input} | preselect: {args.preselect} | repeats: {args.n_time}")
    print(f"Seeds: {args.random_state} through {args.random_state + args.n_time - 1}")

    df = read_dataset(args.input)
    X, y = prep_dataset(df)
    initial_count = X.shape[1]
    X = preselect_motor_and_sensory(X, method=args.preselect)
    expected_count = 132 if args.preselect == "both" else 66
    if initial_count != 132:
        raise ValueError(
            f"Expected 132 original predictors, found {initial_count}. "
            "Check the predictor columns in the input dataset."
        )
    if X.shape[1] != expected_count:
        raise ValueError(
            f"Preselection '{args.preselect}' must retain {expected_count} predictors, "
            f"but produced {X.shape[1]}. Check bilateral predictor pairs "
            "and preselect_predictor.py."
        )
    print("\nX shape after preselection:", X.shape)
    pd.DataFrame({"Predictor": X.columns}).to_csv(
        out_dir / "preselected_predictors.csv", index=False)

    all_fold_results = []
    all_repeat_metrics = []
    all_fold_predictors = []
    all_oof = []
    all_run_bootstraps = []
    all_feature_frequencies = []

    for i in range(args.n_time):
        repeat = i + 1
        seed = args.random_state + i
        run_dir = out_dir / f"repeat_{repeat:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#' * 72}\nREPEAT {repeat}/{args.n_time}  random_state={seed}\n{'#' * 72}", flush=True)

        (results_df, selection_frequency, selected_features_each_fold,
         bootstrap_ci_df, oof_df) = evaluate_nested_cv(
            X, y, n_splits=args.n_splits, random_state=seed,
            out_dir=run_dir, bootstrap_samples=args.bootstrap_samples)

        # Keep full output for each run, as in the original script.
        results_df.to_csv(run_dir / "nested_cv_results.csv", index=False)
        selection_frequency.to_csv(run_dir / "feature_selection_frequency.csv", index=False)
        fold_rows = [
            {"Fold": fold, "Rank": rank, "Predictor": feature}
            for fold, features in enumerate(selected_features_each_fold, start=1)
            for rank, feature in enumerate(features, start=1)
        ]
        pd.DataFrame(fold_rows, columns=["Fold", "Rank", "Predictor"]).to_csv(
            run_dir / "selected_predictors_by_fold.csv", index=False)

        folds = results_df.copy()
        folds.insert(0, "Random_state", seed)
        folds.insert(0, "Repeat", repeat)
        all_fold_results.append(folds)

        # Use pooled OOF metrics for each repetition, not the mean of fold metrics.
        # Each patient has exactly one held-out prediction per repetition.
        run_scores = bootstrap_ci_df.set_index("Metric")["Estimate"]
        all_repeat_metrics.append({"Repeat": repeat, "Random_state": seed,
                                   **{m: float(run_scores[m]) for m in METRIC_COLUMNS}})
        all_run_bootstraps.append(bootstrap_ci_df.assign(Repeat=repeat, Random_state=seed))
        all_oof.append(oof_df.assign(Repeat=repeat, Random_state=seed))
        all_fold_predictors.extend({"Repeat": repeat, "Random_state": seed, **row}
                                   for row in fold_rows)
        all_feature_frequencies.append(selection_frequency.assign(
            Repeat=repeat, Random_state=seed))

        # Checkpoint aggregate results after every completed run.
        pd.concat(all_fold_results, ignore_index=True).to_csv(
            out_dir / "all_fold_results.csv", index=False)
        pd.DataFrame(all_repeat_metrics).to_csv(
            out_dir / "repeat_metrics.csv", index=False)

    repeats_df = pd.DataFrame(all_repeat_metrics)
    summary = summarize_repeated_metrics(
        repeats_df, n_bootstrap=args.bootstrap_samples,
        random_state=args.random_state + args.n_time)
    summary.to_csv(out_dir / "metrics_mean_95CI.csv", index=False)

    combined_oof = pd.concat(all_oof, ignore_index=True)
    combined_oof.to_csv(out_dir / "all_oof_predictions.csv", index=False)
    plot_repeated_calibration(combined_oof, out_dir,
                              n_bins=args.calibration_bins,
                              n_bootstrap=args.bootstrap_samples,
                              random_state=args.random_state + args.n_time + 1)
    pd.concat(all_run_bootstraps, ignore_index=True).to_csv(
        out_dir / "all_run_bootstrap_95CI.csv", index=False)
    pd.concat(all_feature_frequencies, ignore_index=True).to_csv(
        out_dir / "all_feature_selection_frequency.csv", index=False)
    pd.DataFrame(all_fold_predictors,
                 columns=["Repeat", "Random_state", "Fold", "Rank", "Predictor"]
                 ).to_csv(out_dir / "all_selected_predictors_by_fold.csv", index=False)

    selected = pd.DataFrame(all_fold_predictors)
    feature_summary = (selected.groupby("Predictor", as_index=False)
                       .size().rename(columns={"size": "Selected_folds"})
                       .sort_values(["Selected_folds", "Predictor"],
                                    ascending=[False, True]))
    feature_summary["Selection_percentage"] = (
        feature_summary["Selected_folds"] / (args.n_splits * args.n_time) * 100
    )
    feature_summary.to_csv(out_dir / "feature_selection_frequency_overall.csv", index=False)

    settings = {"input": args.input, "preselect": args.preselect,
                "out_dir": str(out_dir), "n_splits": args.n_splits,
                "initial_random_state": args.random_state, "n_time": args.n_time,
                "n_patients": len(y), "n_predictors_after_preselection": X.shape[1],
                "bootstrap_samples": args.bootstrap_samples,
                "calibration_bins": args.calibration_bins,
                "aggregate_CI_method": "Percentile bootstrap of repetition-level pooled OOF metric means"}
    pd.DataFrame(settings.items(), columns=["Setting", "Value"]).to_csv(
        out_dir / "run_settings.csv", index=False)

    print("\n" + "=" * 72)
    print("MEAN PERFORMANCE ACROSS NESTED-CV REPETITIONS (95% BOOTSTRAP CI)")
    print("=" * 72)
    print(summary.round(4).to_string(index=False))
    if args.n_time == 1:
        print("NOTE: n_time=1; between-repeat confidence intervals are undefined.")
    print(f"\nResults saved in: {out_dir}")
    print("Summary: metrics_mean_95CI.csv; run metrics: repeat_metrics.csv")


if __name__ == "__main__":
    main()
