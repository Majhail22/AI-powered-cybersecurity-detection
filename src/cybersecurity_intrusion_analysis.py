"""
Cybersecurity Intrusion Analysis
Refactored from the notebook `Tanish_revised.ipynb` into a GitHub-ready script.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 120)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    if confusion.empty:
        return np.nan
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2corr / denom)


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    df_temp = pd.DataFrame({"cat": categories, "val": measurements}).dropna()
    if df_temp.empty:
        return np.nan
    groups = df_temp.groupby("cat")["val"]
    grand_mean = df_temp["val"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups)
    ss_total = ((df_temp["val"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(np.sqrt(ss_between / ss_total))


def save_fig(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def load_and_prepare_data(data_path: Path) -> tuple[pd.DataFrame, str, list[str], list[str], list[str], list[str]]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path.resolve()}\n"
            "Place the CSV file in the data folder or pass --data with the correct path."
        )

    df = pd.read_csv(data_path).copy()
    target_col = "attack_detected"
    id_cols_to_drop = ["session_id"]

    for col in id_cols_to_drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' was not found in the dataset.")

    if df[target_col].dtype == "object":
        unique_target = sorted(df[target_col].dropna().astype(str).unique().tolist())
        if set(unique_target) <= {"0", "1", "False", "No", "True", "Yes"}:
            mapping = {"0": 0, "1": 1, "No": 0, "Yes": 1, "False": 0, "True": 1}
            df[target_col] = df[target_col].astype(str).map(mapping)

    if "attack_label" not in df.columns:
        df["attack_label"] = df[target_col].map({0: "No Attack", 1: "Attack"}).fillna("Unknown")

    feature_df = df.drop(columns=["attack_label"], errors="ignore")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_feature_cols = [col for col in numeric_cols if col != target_col]
    categorical_feature_cols = [col for col in categorical_cols if col != target_col]

    return df, target_col, numeric_cols, categorical_cols, numeric_feature_cols, categorical_feature_cols


def descriptive_stats(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], output_dir: Path) -> None:
    num_stats = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()
    cat_cols = categorical_cols + (["attack_label"] if "attack_label" in df.columns and "attack_label" not in categorical_cols else [])
    cat_stats = df[cat_cols].describe().T if cat_cols else pd.DataFrame()
    num_stats.to_csv(output_dir / "numeric_summary.csv")
    cat_stats.to_csv(output_dir / "categorical_summary.csv")


def missing_value_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    missing_summary = pd.DataFrame(
        {"missing_count": missing_count, "missing_percent": missing_percent}
    ).sort_values(by="missing_count", ascending=False)
    missing_summary.to_csv(output_dir / "missing_summary.csv")

    missing_only = missing_summary[missing_summary["missing_count"] > 0].reset_index().rename(columns={"index": "column"})
    if not missing_only.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=missing_only, x="column", y="missing_count")
        plt.xticks(rotation=45, ha="right")
        plt.title("Number of missing values in each column")
        save_fig(output_dir, "missing_values_bar.png")

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna().T, cbar=False)
    plt.xlabel("Row number")
    plt.ylabel("Columns")
    plt.title("Missing-value heatmap")
    save_fig(output_dir, "missing_values_heatmap.png")


def compare_encryption_imputation(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame | None:
    col = "encryption_used"
    if col not in df.columns or df[col].dropna().empty:
        return None

    observed = df.loc[df[col].notna(), col].copy()
    rng = np.random.default_rng(42)
    mask_idx = observed.sample(frac=0.2, random_state=42).index
    eval_df = df[[col]].copy()
    true_values = eval_df.loc[mask_idx, col].copy()
    eval_df.loc[mask_idx, col] = np.nan

    results = []

    # Mode imputation
    mode_imputer = SimpleImputer(strategy="most_frequent")
    mode_filled = pd.Series(mode_imputer.fit_transform(eval_df[[col]]).ravel(), index=eval_df.index)
    results.append(
        {"method": "most_frequent", "accuracy": accuracy_score(true_values.astype(str), mode_filled.loc[mask_idx].astype(str))}
    )

    # Constant imputation
    const_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
    const_filled = pd.Series(const_imputer.fit_transform(eval_df[[col]]).ravel(), index=eval_df.index)
    results.append(
        {"method": "constant_unknown", "accuracy": accuracy_score(true_values.astype(str), const_filled.loc[mask_idx].astype(str))}
    )

    # Random observed-value imputation
    observed_values = observed.astype(str).tolist()
    random_fill = eval_df[col].copy()
    random_fill.loc[random_fill.isna()] = rng.choice(observed_values, size=random_fill.isna().sum(), replace=True)
    results.append(
        {"method": "random_observed", "accuracy": accuracy_score(true_values.astype(str), random_fill.loc[mask_idx].astype(str))}
    )

    results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    results_df.to_csv(output_dir / "encryption_imputation_comparison.csv", index=False)
    return results_df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if cat_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    if num_cols:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    return df


def outlier_analysis(df: pd.DataFrame, numeric_feature_cols: list[str], output_dir: Path) -> dict[str, list[int]]:
    outlier_summary = []
    outlier_idx: dict[str, list[int]] = {}

    for col in numeric_feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_idx[col] = df.index[mask].tolist()
        outlier_summary.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_limit": lower,
                "upper_limit": upper,
                "outlier_count": int(mask.sum()),
            }
        )

    pd.DataFrame(outlier_summary).sort_values(by="outlier_count", ascending=False).to_csv(
        output_dir / "outlier_summary.csv", index=False
    )

    threshold = 3.5
    significance_results = []
    for col in numeric_feature_cols:
        values = df[col].astype(float)
        median = values.median()
        mad = (values - median).abs().median()
        mild_or_borderline = []

        for idx in outlier_idx.get(col, []):
            val = values.loc[idx]
            if mad > 0:
                mod_z = 0.6745 * (val - median) / mad
            else:
                std = values.std(ddof=0)
                mod_z = 0 if std == 0 else (val - median) / std

            if abs(mod_z) < threshold:
                mild_or_borderline.append(int(idx))

        significance_results.append(
            {
                "column": col,
                "total_iqr_outliers": len(outlier_idx.get(col, [])),
                "mild_or_borderline_outliers": len(mild_or_borderline),
                "strong_outliers": len(outlier_idx.get(col, [])) - len(mild_or_borderline),
            }
        )

    pd.DataFrame(significance_results).to_csv(output_dir / "outlier_significance_summary.csv", index=False)
    return outlier_idx


def exploratory_plots(df: pd.DataFrame, numeric_feature_cols: list[str], categorical_feature_cols: list[str], output_dir: Path) -> None:
    count_vars = []
    for col in categorical_feature_cols + (["attack_label"] if "attack_label" in df.columns else []):
        if df[col].nunique(dropna=False) <= 12:
            count_vars.append(col)

    if count_vars:
        n_cols = 2
        n_rows = int(np.ceil(len(count_vars) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, count_vars):
            order = df[col].astype(str).value_counts().index
            sns.countplot(data=df, x=col, order=order, ax=ax)
            ax.set_title(f"Count plot of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.tick_params(axis="x", rotation=45)
        for ax in axes[len(count_vars):]:
            ax.axis("off")
        save_fig(output_dir, "count_plots.png")

    num_matrix = numeric_feature_cols[: min(4, len(numeric_feature_cols))]
    if num_matrix:
        fig, axes = plt.subplots(len(num_matrix), 1, figsize=(10, 4 * len(num_matrix)))
        if len(num_matrix) == 1:
            axes = [axes]

        for ax, col in zip(axes, num_matrix):
            data = df[col].dropna()
            desc = data.describe()
            skewness = data.skew()
            kurtosis = data.kurtosis()

            data.hist(bins=30, ax=ax, edgecolor="black", alpha=0.7)
            ax2 = ax.twinx()
            data.plot(kind="kde", ax=ax2, linewidth=2)

            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax2.set_ylabel("Density")

            summary_text = (
                f"count={int(desc['count'])}\n"
                f"mean={desc['mean']:.2f}\n"
                f"std={desc['std']:.2f}\n"
                f"skew={skewness:.2f}\n"
                f"kurt={kurtosis:.2f}"
            )
            ax.text(0.98, 0.95, summary_text, transform=ax.transAxes, ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        save_fig(output_dir, "numeric_distributions.png")

    cat_plot_cols = [col for col in categorical_feature_cols if df[col].nunique() <= 8][:4]
    if cat_plot_cols:
        n = len(cat_plot_cols)
        n_cols = 2
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, cat_plot_cols):
            counts = df[col].value_counts()
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.82,
            )
            centre_circle = plt.Circle((0, 0), 0.60, fc="white")
            ax.add_artist(centre_circle)
            ax.set_title(f"Distribution of {col}")

        for ax in axes[len(cat_plot_cols):]:
            ax.axis("off")
        save_fig(output_dir, "donut_charts.png")


def mixed_type_association_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Numeric-numeric
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        corr.to_csv(output_dir / "pearson_correlation.csv")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Pearson correlation heatmap")
        save_fig(output_dir, "pearson_correlation_heatmap.png")

    # Categorical-categorical
    if len(categorical_cols) >= 2:
        cramers = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
        for c1 in categorical_cols:
            for c2 in categorical_cols:
                cramers.loc[c1, c2] = cramers_v(df[c1], df[c2])
        cramers.to_csv(output_dir / "cramers_v_matrix.csv")

    # Numeric-categorical
    if num_cols and categorical_cols:
        eta = pd.DataFrame(index=num_cols, columns=categorical_cols, dtype=float)
        for n in num_cols:
            for c in categorical_cols:
                eta.loc[n, c] = correlation_ratio(df[c], df[n])
        eta.to_csv(output_dir / "correlation_ratio_matrix.csv")


def train_and_evaluate_models(df: pd.DataFrame, target_col: str, output_dir: Path) -> pd.DataFrame:
    X = df.drop(columns=[target_col, "attack_label"], errors="ignore")
    y = df[target_col]

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model_numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    model_categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), model_numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), model_categorical_cols),
        ],
        remainder="drop",
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    roc_curves = {}
    pr_curves = {}
    best_estimators = {}

    for model_name, model in models.items():
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        best_estimators[model_name] = pipeline

        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline.named_steps["model"], "decision_function"):
            y_score = pipeline.decision_function(X_test)
        else:
            y_score = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        if y_score is not None:
            roc_auc = roc_auc_score(y_test, y_score)
            pr_auc = average_precision_score(y_test, y_score)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
            roc_curves[model_name] = (fpr, tpr)
            pr_curves[model_name] = (recall_curve, precision_curve)
        else:
            roc_auc = np.nan
            pr_auc = np.nan

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        params = pipeline.named_steps["model"].get_params()
        key_params = {
            k: v
            for k, v in params.items()
            if k in ["C", "max_depth", "n_estimators", "kernel", "n_neighbors", "max_features"]
        }

        results.append(
            {
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "BestParams": str(key_params),
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False).reset_index(drop=True)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)

    if roc_curves:
        plt.figure(figsize=(8, 6))
        for name, (fpr, tpr) in roc_curves.items():
            plt.plot(fpr, tpr, label=name)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curves for classification models")
        plt.legend()
        save_fig(output_dir, "roc_curves.png")

    if pr_curves:
        plt.figure(figsize=(8, 6))
        for name, (recall_curve, precision_curve) in pr_curves.items():
            plt.plot(recall_curve, precision_curve, label=name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curves for classification models")
        plt.legend()
        save_fig(output_dir, "precision_recall_curves.png")

    best_model_name = results_df.loc[0, "Model"]
    best_model = best_estimators[best_model_name]
    best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion matrix - {best_model_name}")
    save_fig(output_dir, "best_model_confusion_matrix.png")

    # Optional SHAP explainability
    try:
        import shap  # type: ignore

        if best_model_name in ["Decision Tree", "Random Forest"]:
            prep = best_model.named_steps["prep"]
            model = best_model.named_steps["model"]

            X_test_transformed = prep.transform(X_test)
            if hasattr(X_test_transformed, "toarray"):
                X_test_transformed = X_test_transformed.toarray()

            feature_names = model_numeric_cols.copy()
            if model_categorical_cols:
                ohe = prep.named_transformers_["cat"]
                feature_names += list(ohe.get_feature_names_out(model_categorical_cols))

            sample_size = min(300, X_test_transformed.shape[0])
            sample_idx = np.random.default_rng(42).choice(
                X_test_transformed.shape[0], size=sample_size, replace=False
            )
            X_sample = X_test_transformed[sample_idx]

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            plt.figure()
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_sample, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            save_fig(output_dir, "shap_summary.png")
    except Exception:
        pass

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cybersecurity intrusion analysis pipeline")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/cybersecurity_intrusion_data.csv"),
        help="Path to the input CSV dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory to save generated results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df, target_col, numeric_cols, categorical_cols, numeric_feature_cols, categorical_feature_cols = load_and_prepare_data(args.data)
    descriptive_stats(df, numeric_cols, categorical_cols, args.output)
    missing_value_analysis(df, args.output)
    compare_encryption_imputation(df, args.output)
    df = fill_missing_values(df)
    outlier_analysis(df, numeric_feature_cols, args.output)
    exploratory_plots(df, numeric_feature_cols, categorical_feature_cols, args.output)
    mixed_type_association_analysis(df, args.output)
    results_df = train_and_evaluate_models(df, target_col, args.output)

    print("Analysis complete.")
    print(f"Best model: {results_df.loc[0, 'Model']}")
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
