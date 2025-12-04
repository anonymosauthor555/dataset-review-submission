import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import KFold
from autogluon.tabular import TabularPredictor
from config.config_loader import load_config


# =============================================================
#                    CONFIGURATION & DATA LOADING
# =============================================================
FILE_NAME = "whole_feature_120_1_new.csv"
RESULT_CSV = "model_results_final111.csv"
SAVE_DIR = "./autogluon_models_test_EEEE"
os.makedirs(SAVE_DIR, exist_ok=True)

label_col = "label"
participant_col = "participant"
n_splits = 10
n_jobs = 20


# -------- Load feature dataset -------- #
df = pd.read_csv(FILE_NAME)

# Remove raw eye-position features measured in DACSmm (low usability, redundant)
df = df.drop(columns=[c for c in df.columns if "DACSmm" in c])


# -------- Load participant configuration -------- #
CONFIG_DATA = load_config(os.path.join("config", "feature_engineering_config.json"))
NC_number = CONFIG_DATA["NC_number"]
PD_number = CONFIG_DATA["PD_number"]

nc_subjects = [f"NC P{i}" for i in range(1, NC_number + 1)]
pd_subjects = [f"PD P{i}" for i in range(1, PD_number + 1)]
all_subjects = nc_subjects + pd_subjects

# Remove subjects with high eye-tracking data loss
exclude_subjects1 = [
    "NC P16","NC P26","NC P8",
    "PD P18","PD P1","PD P2","PD P4","PD P21","PD P26",
    "NC P18"
]
all_subjects = [s for s in all_subjects if s not in exclude_subjects1]


# =============================================================
#                    FEATURE GROUP DEFINITION
# =============================================================
# Group features by sensing modality (steering, pedal, speed, eye/head)
groups = {
    "steering_all": [
        c for c in df.columns if c.startswith(
            ("STATSPNEW_steering","STATSP_steering","FFT_steering",
             "WVL_steering","NewPedal_steering","LAST_steer"))
    ],
    "pedal_all": [
        c for c in df.columns if c.startswith(
            ("STATSP_throttle","STATSPNEW_throttle","FFT_throttle",
             "WVL_throttle","LAST_throttle",
             "STATSP_brake","STATSPNEW_brake","FFT_brake","WVL_brake","LAST_brake"))
    ],
    "speed_all": [
        c for c in df.columns if c.startswith(("STATSS","STATSSNEW_","NewSpeed"))
    ],
    "evemovement_all": [
        c for c in df.columns if c.startswith(
            ("STATSE_","STATSENEW_","GE_","STATSH_","STATSHNEW_"))
    ]
}

print("\n=========== Feature Group Summary ===========")
for g, cols in groups.items():
    print(f"{g}: {len(cols)} features")
print("=============================================\n")


# =============================================================
#                    FEATURE GROUP COMBINATIONS
# =============================================================
# Build all combinations of ≥4 sensing modalities
combo_groups = {}
group_names = list(groups.keys())

for r in range(4, len(group_names) + 1):
    for combo in itertools.combinations(group_names, r):
        combo_name = "_AND_".join(combo)
        combo_groups[combo_name] = [c for g in combo for c in groups[g]]

print(f"Total {len(combo_groups)} feature combinations.\n")


# =============================================================
#      CORE: Single-Fold Training using AutoGluon Tabular
# =============================================================
def run_fold(train_index, test_index, subject_list, df, cols, label_col, n_jobs):
    """Leave-subject-out learning: train on subjects in train_index and test on others"""
    
    train_subjects = [subject_list[i] for i in train_index]
    test_subjects = [subject_list[i] for i in test_index]

    df_train = df[df["participant"].isin(train_subjects)][[label_col] + cols]
    df_test = df[df["participant"].isin(test_subjects)][[label_col] + cols]

    if df_train.empty or df_test.empty:
        return None

    # Create unique folder for AutoGluon model serialization
    fold_model_path = os.path.join(SAVE_DIR, f"fold_{os.getpid()}_{len(train_subjects)}")
    os.makedirs(fold_model_path, exist_ok=True)

    # Train AutoGluon using default model ensemble (no stacking/bagging)
    predictor = TabularPredictor(
        label=label_col,
        eval_metric="roc_auc",
        path=fold_model_path,
    ).fit(
        train_data=df_train,
        num_stack_levels=0,
        num_bag_folds=0,
        ag_args_fit={"num_cpus": n_jobs},
        holdout_frac=0,
        verbosity=0
    )

    model_list = predictor.leaderboard(silent=True)["model"].tolist()

    results = {}

    for model_name in model_list:

        preds = predictor.predict(df_test, model=model_name)
        proba = predictor.predict_proba(df_test, model=model_name).values
        y_test = df_test[label_col].values

        # Performance metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        try:
            auc = roc_auc_score(y_test, proba[:, 1])
        except:
            auc = np.nan

        # Sensitivity & specificity
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)

        results[model_name] = {
            "acc": acc,
            "f1": f1,
            "auc": auc,
            "sens": sens,
            "spec": spec
        }

    return {
        "fold_model_scores": results,
        "test_subjects": test_subjects,
    }


# =============================================================
#                 MAIN LOOP: CV OVER GROUPS
# =============================================================
all_results = []

for feature_group_name, cols in tqdm(combo_groups.items(), desc="Feature Group Loop"):

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(cv.split(all_subjects))

    fold_outputs = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(
            train_idx, test_idx, all_subjects, df, cols, label_col, n_jobs
        )
        for train_idx, test_idx in folds
    )

    fold_outputs = [fr for fr in fold_outputs if fr is not None]

    model_names = fold_outputs[0]["fold_model_scores"].keys()

    for model_name in model_names:

        acc_list, f1_list, auc_list, sens_list, spec_list = [], [], [], [], []

        for fr in fold_outputs:
            ms = fr["fold_model_scores"][model_name]
            acc_list.append(ms["acc"])
            f1_list.append(ms["f1"])
            auc_list.append(ms["auc"])
            sens_list.append(ms["sens"])
            spec_list.append(ms["spec"])

        result_row = {
            "model_name": model_name,
            "feature_group": feature_group_name,
            "accuracy_mean": np.mean(acc_list),
            "f1_macro_mean": np.mean(f1_list),
            "auroc": np.mean(auc_list),
            "auroc_std": np.nanstd(auc_list),
            "sensitivity_mean": np.mean(sens_list),
            "specificity_mean": np.mean(spec_list),
            "acc_each_fold": acc_list,
            "auc_each_fold": auc_list,
        }

        all_results.append(result_row)


# Save final aggregated table
results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULT_CSV, index=False)

print(f"\n======== ALL RESULTS SAVED TO {RESULT_CSV} ========\n")
