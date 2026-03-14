import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score, KFold, cross_validate
import xgboost as xgb
import shap
from scipy.stats import spearmanr
from tabulate import tabulate

# ── Data Loading ──────────────────────────────────────────────────────────────
data = pd.read_excel('Data_Desalination.xlsx', header=3)
data = data.iloc[1:, 3:]

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna()
print("Data shape after cleaning:", data.shape)
data.to_csv('data.csv', index=False)

# ── Target Column Detection ───────────────────────────────────────────────────
if 'Permeate conductivity' in data.columns:
    target_col = 'Permeate conductivity'
else:
    candidates = [c for c in data.columns
                  if 'conductivity' in c.lower() and 'permeate' in c.lower()]
    if candidates:
        target_col = candidates[0]
    else:
        candidates = [c for c in data.columns if 'conductivity' in c.lower()]
        if candidates:
            target_col = candidates[0]
        else:
            raise ValueError("Could not find a suitable target column.")

print(f"Target column: {target_col}")

X = data.drop(columns=[target_col])
y = data[target_col]
print(f"Feature shape: {X.shape} | Target shape: {y.shape}")

# ── Cross-Validation Setup ────────────────────────────────────────────────────
N_SPLITS    = 5
N_FEATURES  = 5
RANDOM_SEED = 42

outer_cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# ── CV Evaluation Helper ──────────────────────────────────────────────────────
def evaluate_cv(X_sub, y, cv, label=""):
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)

    cv_results = cross_validate(
        rf, X_sub, y,
        cv=cv,
        scoring='r2',
        return_train_score=True,
        n_jobs=-1
    )

    fold_r2s   = cv_results['test_score']
    mean_r2    = fold_r2s.mean()
    std_r2     = fold_r2s.std()
    train_mean = cv_results['train_score'].mean()

    print(
        f"  [{label}] folds={N_SPLITS} | "
        f"test R2: {mean_r2:.4f} +/- {std_r2:.4f} | "
        f"train R2 mean: {train_mean:.4f}"
    )

    return {
        "mean_r2"   : mean_r2,
        "std_r2"    : std_r2,
        "fold_r2s"  : fold_r2s.tolist()
    }

# ── Feature Selection Methods ─────────────────────────────────────────────────
def rf_feature_selection(X, y, n):
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X, y)
    idx = np.argsort(model.feature_importances_)[::-1]
    return X.columns[idx][:n].tolist(), model

def xgb_feature_selection(X, y, n):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_SEED,
        verbosity=0
    )
    model.fit(X, y)
    idx = np.argsort(model.feature_importances_)[::-1]
    return X.columns[idx][:n].tolist(), model

def rf_shap_feature_selection(X, y, n):
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X, y)
    shap_vals = shap.TreeExplainer(model).shap_values(X)
    importance = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(importance)[::-1]
    return X.columns[idx][:n].tolist(), model

def xgb_shap_feature_selection(X, y, n):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_SEED,
        verbosity=0
    )
    model.fit(X, y)
    shap_vals = shap.TreeExplainer(model).shap_values(X)
    importance = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(importance)[::-1]
    return X.columns[idx][:n].tolist(), model

def feature_agglomeration_selection(X, y, n):
    n_clusters = min(5, X.shape[1])
    fa = FeatureAgglomeration(n_clusters=n_clusters).fit(X)
    feat_var = [
        (X.columns[i], X.iloc[:, i].var(), fa.labels_[i])
        for i in range(X.shape[1])
    ]
    feat_var.sort(key=lambda t: t[1], reverse=True)

    selected, seen_clusters = [], set()
    for feat, _, cluster in feat_var:
        if len(selected) >= n:
            break
        if cluster not in seen_clusters:
            selected.append(feat)
            seen_clusters.add(cluster)

    remaining = [f for f, _, _ in feat_var if f not in selected]
    selected += remaining[:n - len(selected)]
    return selected[:n], None

def hvgs_selection(X, y, n):
    return X.var().sort_values(ascending=False).index[:n].tolist(), None

def spearman_selection(X, y, n):
    corrs = [
        (col, abs(spearmanr(X[col], y).correlation))
        for col in X.columns
    ]
    corrs.sort(key=lambda t: t[1], reverse=True)
    return [c for c, _ in corrs[:n]], None

# ── Main Experiment Loop ──────────────────────────────────────────────────────
methods = {
    'RF'      : rf_feature_selection,
    'XGB'     : xgb_feature_selection,
    'RF-SHAP' : rf_shap_feature_selection,
    'XGB-SHAP': xgb_shap_feature_selection,
    'FA'      : feature_agglomeration_selection,
    'HVGS'    : hvgs_selection,
    'Spearman': spearman_selection,
}

results = {}

for name, method in methods.items():
    print(f"\n{'='*60}")
    print(f" Method: {name}")
    print(f"{'='*60}")

    # Set 1: top-5 features from full feature space
    top5, _ = method(X, y, N_FEATURES)
    print(f"  Top-5 features : {top5}")
    cv5_stats = evaluate_cv(X[top5], y, outer_cv, label=f"{name} | top-5")

    # Set 2: top-4 features after dropping the #1 ranked feature
    top_feature = top5[0]
    X_reduced   = X.drop(columns=[top_feature])
    top4, _     = method(X_reduced, y, N_FEATURES - 1)
    print(f"  Dropped feature : {top_feature}")
    print(f"  Top-4 features  : {top4}")
    cv4_stats = evaluate_cv(X_reduced[top4], y, outer_cv, label=f"{name} | top-4")

    results[name] = {
        "top5_features"  : top5,
        "top4_features"  : top4,
        "dropped_feature": top_feature,
        "cv5_mean_r2"    : cv5_stats["mean_r2"],
        "cv5_std_r2"     : cv5_stats["std_r2"],
        "cv4_mean_r2"    : cv4_stats["mean_r2"],
        "cv4_std_r2"     : cv4_stats["std_r2"],
        "cv5_fold_r2s"   : cv5_stats["fold_r2s"],
        "cv4_fold_r2s"   : cv4_stats["fold_r2s"],
    }

# ── Summary Table ─────────────────────────────────────────────────────────────
summary_rows = []
for method_name, r in results.items():
    summary_rows.append({
        "Method"         : method_name,
        "CV5 R2 mean"    : f"{r['cv5_mean_r2']:.4f}",
        "CV5 R2 std"     : f"{r['cv5_std_r2']:.4f}",
        "CV4 R2 mean"    : f"{r['cv4_mean_r2']:.4f}",
        "CV4 R2 std"     : f"{r['cv4_std_r2']:.4f}",
        "Dropped Feature": r['dropped_feature'],
        "Top-5 Features" : ' | '.join(r['top5_features']),
        "Top-4 Features" : ' | '.join(r['top4_features']),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('result.csv', index=False)

print("\n\nSummary Table")
print("=" * 80)
print(tabulate(summary_rows, headers="keys", tablefmt="github"))
print("\nResults saved to result.csv")
