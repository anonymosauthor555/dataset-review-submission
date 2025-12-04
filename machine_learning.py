import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import KFold, LeaveOneOut

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from config.config_loader import load_config


# ====================== Load and preprocess feature dataset ====================== #
file_name = "whole_feature_120_1_new.csv"
df = pd.read_csv(file_name)

# Drop raw-eye-position features measured in DACSmm (highly correlated)
columns_to_remove = [col for col in df.columns if 'DACSmm' in col]
df = df.drop(columns=columns_to_remove)

results_file_path = "model_results_new_11111.csv"

cv_strategy = "kfold"
n_splits = 10
label_col = "label"


# ====================== Model builder with unified preprocessing ====================== #
def make_model_pipeline(model_name: str):
    """
    Build sklearn model pipeline with:
        - Missing-value imputation
        - Feature standardization
        - Classifier
    """
    model_name = model_name.lower()

    if model_name == "logistic regression":
        clf = LogisticRegression(
            penalty="l1", solver="saga", max_iter=10000, class_weight="balanced"
        )
    elif model_name == "elasticnet":
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            max_iter=10000, class_weight="balanced"
        )
    elif model_name == "ridge":
        clf = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=10000, class_weight="balanced"
        )
    elif model_name == "svm":
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced")
    elif model_name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256,256,256), max_iter=5000)
    elif model_name == "random forest":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
    elif model_name == "gradient boosting":
        clf = GradientBoostingClassifier(n_estimators=200)
    elif model_name == "xgboost":
        clf = XGBClassifier(
            n_estimators=200, eval_metric="logloss", use_label_encoder=False
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    return model


# ====================== Feature grouping by modality ====================== #
# Used for evaluating different sensor combinations
groups = {
    "steering_all": [
        col for col in df.columns
        if col.startswith(("STATSPNEW_steering", "STATSP_steering",
                           "FFT_steering", "WVL_steering",
                           "NewPedal_steering", "LAST_steer"))
    ],
    "pedal_all": [
        col for col in df.columns
        if col.startswith(("STATSP_throttle","STATSPNEW_throttle","FFT_throttle",
                           "WVL_throttle","LAST_throttle",
                           "STATSP_brake","STATSPNEW_brake","FFT_brake",
                           "WVL_brake","LAST_brake"))
    ],
    "speed_all": [
        col for col in df.columns
        if col.startswith(("STATSS","STATSSNEW_","NewSpeed"))
    ],
    "evemovement_all": [
        col for col in df.columns
        if col.startswith(("STATSE_","STATSENEW_","GE_","STATSH_","STATSHNEW_"))
    ]
}

for group_name, cols in groups.items():
    print(f"{group_name}: {len(cols)} parameters")


# ====================== Subject metadata loaded from config ====================== #
CONFIG_DATA = load_config(os.path.join("config", "feature_engineering_config.json"))
NC_number = CONFIG_DATA["NC_number"]
PD_number = CONFIG_DATA["PD_number"]

nc_subjects_all = [f"NC P{i}" for i in range(1, NC_number+1)]
pd_subjects_all = [f"PD P{i}" for i in range(1, PD_number+1)]
all_subjects = nc_subjects_all + pd_subjects_all

# Exclude subjects with >50% gaze loss
exclude_subjects1 = [
    "NC P16","NC P26","NC P8","PD P18","PD P1","PD P2",
    "PD P4","PD P21","PD P26","NC P18"
]
all_subjects = [s for s in all_subjects if s not in exclude_subjects1]


# ====================== Single fold execution function ====================== #
def run_fold(train_index, test_index, all_subjects_filtered, df, cols, label_col, model_name):
    """
    Train on subjects in train_index and test on subjects in test_index.
    Leave-subject-out style: no sample leakage.
    """
    train_subjects = [all_subjects_filtered[i] for i in train_index]
    test_subjects = [all_subjects_filtered[i] for i in test_index]

    df_train = df[df["participant"].isin(train_subjects)].copy()
    df_test = df[df["participant"].isin(test_subjects)].copy()

    if df_train.empty or df_test.empty:
        return None

    X_train = df_train[cols].copy()
    y_train = df_train[label_col].copy()
    X_test = df_test[cols].copy()
    y_test = df_test[label_col].copy()

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return None

    model = make_model_pipeline(model_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    result = {
        "acc": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "y_true": y_test.tolist(),
        "y_score": y_score.tolist(),
        "subject": test_subjects,
    }

    # Binary metrics
    if len(np.unique(y_test)) == 2:
        sens = recall_score(y_test, y_pred, pos_label=1)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        result["sens"] = sens
        result["spec"] = spec

        try:
            result["auc"] = roc_auc_score(y_test, y_score[:, 1])
        except:
            result["auc"] = np.nan

    return result


# ====================== Build feature combinations ====================== #
# Combine 4 modalities: evaluate all ≥4 combinations
combo_groups = {}
group_names = list(groups.keys())

for r in range(4, len(group_names) + 1):
    for combo in itertools.combinations(group_names, r):
        combo_name = "_AND_".join(combo)
        combo_groups[combo_name] = [col for g in combo for col in groups[g]]


# ====================== Optional ROC plot function (unused in pipeline) ====================== #
def plot_mean_roc(y_true_all, y_score_all, auc_list, title="Early Warning", save_path=None):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for _ in auc_list:
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all[:, 1])
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)

    plt.figure(figsize=(5,5))
    plt.plot(mean_fpr, mean_tpr, lw=2)
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1), alpha=0.3)
    plt.plot([0, 1], [0, 1], "--", color="gray")

    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)
    plt.text(0.7, 0.1, f"AUROC\n{mean_auc:.2f}±{std_auc:.2f}",
             bbox=dict(facecolor="white", edgecolor="black"), fontsize=10)

    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ====================== Main evaluation loop: models × feature sets ====================== #
model_names = [
    "logistic regression", "elasticnet", "ridge", "svm",
    "mlp", "random forest", "gradient boosting", "xgboost"
]
n_jobs = 15

for model_name in model_names:
    results = []

    for name, cols in tqdm(combo_groups.items(), desc="Feature Groups"):
        acc_list, f1_list, sens_list, spec_list, auc_list = [], [], [], [], []
        y_true_all, y_score_all = [], []

        if cv_strategy == "kfold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        folds = []
        for fold_id, (train_index, test_index) in enumerate(cv.split(all_subjects)):
            test_subjects = [all_subjects[i] for i in test_index]
            folds.append((train_index, test_index, test_subjects))

        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(run_fold)(train_index, test_index, all_subjects,
                              df, cols, label_col, model_name)
            for train_index, test_index, test_subjects in folds
        )

        fold_results = [fr for fr in fold_results if fr is not None]

        for fr in fold_results:
            acc_list.append(fr["acc"])
            f1_list.append(fr["f1"])
            y_true_all.extend(fr["y_true"])
            y_score_all.extend(fr["y_score"])

            if "sens" in fr:
                sens_list.append(fr["sens"])
                spec_list.append(fr["spec"])
                auc_list.append(fr["auc"])

        y_true_all = np.array(y_true_all)
        y_score_all = np.array(y_score_all)

        try:
            if len(np.unique(y_true_all)) == 2:
                auroc = roc_auc_score(y_true_all, y_score_all[:, 1])
            else:
                auroc = roc_auc_score(
                    y_true_all, y_score_all,
                    multi_class="ovr", average="macro"
                )
        except:
            auroc = np.nan

        results.append({
            "model_name": model_name,
            "feature_group": name,
            "cv_strategy": cv_strategy,
            "n_splits": n_splits,
            "accuracy_mean": np.mean(acc_list),
            "f1_macro_mean": np.mean(f1_list),
            "auroc": auroc,
            "auroc_std": np.nanstd(auc_list) if len(auc_list) > 0 else np.nan,
            "sensitivity_mean": np.mean(sens_list) if sens_list else np.nan,
            "specificity_mean": np.mean(spec_list) if spec_list else np.nan,
            "acc_each_fold": [round(a, 2) for a in acc_list],
            "auc_each_fold": [round(a, 2) if not np.isnan(a) else np.nan for a in auc_list],
        })

    results_df = pd.DataFrame(results)
    if not os.path.exists(results_file_path):
        results_df.to_csv(results_file_path, mode='w', index=False, header=True)
    else:
        results_df.to_csv(results_file_path, mode='a', index=False, header=False)

    print(f"Results for {model_name} saved to {results_file_path}")
