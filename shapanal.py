import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import r2_score
import xgboost as xgb
from scipy.stats import spearmanr
import shap
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Load the Excel file, skipping the header rows
data = pd.read_excel('Data_Desalination.xlsx', header=3)

# Print original data shape
print(f"\nOriginal data shape (with header rows): {data.shape}")

# Remove the units row and first 3 columns
data = data.iloc[1:, 3:]

# Print data shape after removing units row and first 3 columns
print(f"Data shape after removing units row and first 3 columns: {data.shape}")

print("\nOriginal data types:")
print(data.dtypes)

# Clean the data by replacing commas with dots and converting to float
for col in data.columns:
    if data[col].dtype == 'object':
        # Replace commas with dots (for international number formats)
        data[col] = data[col].astype(str).str.replace(',', '.')
        
        # Convert to float
        data[col] = data[col].astype(float)

print("\nCleaned data types:")
print(data.dtypes)

# Find target column (it may have extra spaces)
target_column = [col for col in data.columns if 'Permeate' in col and 'conductivity' in col][0]
print(f"\nTarget column found: '{target_column}'")

# Split into features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Print shapes of features and target
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes of training and testing sets
print(f"Training features (X_train) shape: {X_train.shape}")
print(f"Testing features (X_test) shape: {X_test.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}\n")

# Dictionary to store all results
results = {
    "method": [],
    "dataset": [],
    "train_r2": [],
    "test_r2": [],
    "full_rankings": []  # Will store FULL feature rankings
}

# Dictionary to store feature rankings
feature_rankings = {}

# Function to add results to our results dictionary
def add_result(method, dataset, train_r2, test_r2, ranked_features=None):
    results["method"].append(method)
    results["dataset"].append(dataset)
    # Format R² scores to 4 decimal places
    results["train_r2"].append(round(train_r2, 4) if train_r2 is not None else None)
    results["test_r2"].append(round(test_r2, 4) if test_r2 is not None else None)
    
    # Add FULL feature rankings as a comma-separated string
    if ranked_features is not None:
        full_rankings = ', '.join(ranked_features)
        results["full_rankings"].append(full_rankings)
    else:
        results["full_rankings"].append("")

# 1. Random Forest
def run_random_forest(X_train, X_test, y_train, y_test, feature_names, dataset_name="Full"):
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get predictions
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    # Calculate R2 scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Store feature rankings
    method_name = f"Random Forest ({dataset_name})"
    ranked_features = feature_importance['Feature'].tolist()
    feature_rankings[method_name] = ranked_features
    
    add_result(method_name, dataset_name, train_r2, test_r2, ranked_features)
    
    return feature_importance, rf

# 2. XGBoost
def run_xgboost(X_train, X_test, y_train, y_test, feature_names, dataset_name="Full"):
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Get predictions
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    
    # Calculate R2 scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Store feature rankings
    method_name = f"XGBoost ({dataset_name})"
    ranked_features = feature_importance['Feature'].tolist()
    feature_rankings[method_name] = ranked_features
    
    add_result(method_name, dataset_name, train_r2, test_r2, ranked_features)
    
    return feature_importance, xgb_model

# 3. Feature Agglomeration
def run_feature_agglomeration(X_train, X_test, y_train, y_test, feature_names, dataset_name="Full"):
    # Number of clusters to create
    n_clusters = max(1, len(feature_names) // 2)  # Using half the number of features
    
    # Apply Feature Agglomeration
    agglomeration = FeatureAgglomeration(n_clusters=n_clusters)
    X_train_agg = agglomeration.fit_transform(X_train)
    X_test_agg = agglomeration.transform(X_test)
    
    # Train Random Forest on agglomerated features for performance evaluation
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_agg, y_train)
    
    # Get predictions
    y_train_pred = rf.predict(X_train_agg)
    y_test_pred = rf.predict(X_test_agg)
    
    # Calculate R2 scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Get feature importance by calculating correlation between original features and clusters
    feature_importance = []
    for i in range(n_clusters):
        for j, feature in enumerate(feature_names):
            corr = np.abs(np.corrcoef(X_train[:, j], X_train_agg[:, i])[0, 1])
            feature_importance.append((feature, i, corr))
    
    # Find top feature for each cluster
    clusters = {}
    for feature, cluster, corr in feature_importance:
        if cluster not in clusters or corr > clusters[cluster][1]:
            clusters[cluster] = (feature, corr)
    
    top_features = [f[0] for f in sorted(clusters.values(), key=lambda x: x[1], reverse=True)]
    
    # Create a full ranking by adding remaining features
    remaining_features = [f for f in feature_names if f not in top_features]
    full_ranking = top_features + remaining_features
    
    # Store feature rankings
    method_name = f"Feature Agglomeration ({dataset_name})"
    feature_rankings[method_name] = full_ranking
    
    add_result(method_name, dataset_name, train_r2, test_r2, full_ranking)
    
    return pd.DataFrame({'Feature': full_ranking}), agglomeration

# 4. Highly Variable Gene Selection (using variance)
def run_hvgs(X_train, X_test, y_train, y_test, feature_names, dataset_name="Full"):
    # Calculate variance of each feature
    variance_selector = VarianceThreshold()
    variance_selector.fit(X_train)
    variances = variance_selector.variances_
    
    # Create a dataframe for feature variance
    feature_variance = pd.DataFrame({
        'Feature': feature_names,
        'Variance': variances
    }).sort_values(by='Variance', ascending=False)
    
    # Get top features
    top_features = feature_variance['Feature'].tolist()
    
    # Use top half of features to train a model for performance evaluation
    top_half_idx = [list(feature_names).index(f) for f in top_features[:len(feature_names)//2]]
    X_train_top = X_train[:, top_half_idx]
    X_test_top = X_test[:, top_half_idx]
    
    # Train Random Forest on top features for performance evaluation
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_top, y_train)
    
    # Get predictions
    y_train_pred = rf.predict(X_train_top)
    y_test_pred = rf.predict(X_test_top)
    
    # Calculate R2 scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Store feature rankings
    method_name = f"HVGS ({dataset_name})"
    feature_rankings[method_name] = top_features
    
    add_result(method_name, dataset_name, train_r2, test_r2, top_features)
    
    return feature_variance, variance_selector

# 5. Spearman Correlation
def run_spearman(X_train, X_test, y_train, y_test, feature_names, dataset_name="Full"):
    # Calculate Spearman correlation between each feature and the target
    correlations = []
    for i, feature in enumerate(feature_names):
        corr, _ = spearmanr(X_train[:, i], y_train)
        correlations.append((feature, abs(corr)))  # Using absolute correlation
    
    # Sort features by correlation
    sorted_corrs = sorted(correlations, key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_corrs]
    
    # Use top half of features to train a model for performance evaluation
    top_half_idx = [list(feature_names).index(f) for f in top_features[:len(feature_names)//2]]
    X_train_top = X_train[:, top_half_idx]
    X_test_top = X_test[:, top_half_idx]
    
    # Train Random Forest on top features for performance evaluation
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_top, y_train)
    
    # Get predictions
    y_train_pred = rf.predict(X_train_top)
    y_test_pred = rf.predict(X_test_top)
    
    # Calculate R2 scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Store feature rankings
    method_name = f"Spearman ({dataset_name})"
    feature_rankings[method_name] = top_features
    
    add_result(method_name, dataset_name, train_r2, test_r2, top_features)
    
    return pd.DataFrame({'Feature': top_features, 'Correlation': [c[1] for c in sorted_corrs]}), None

# 6. SHAP for Random Forest - Updated to remove prediction accuracy
def run_rf_shap(X_train, X_test, y_train, y_test, feature_names, rf_model, dataset_name="Full"):
    # Create explainer
    explainer = shap.TreeExplainer(rf_model)
    
    # Calculate SHAP values on a sample of training data (for computational efficiency)
    sample_size = min(100, X_train.shape[0])
    X_sample = X_train[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    
    # Calculate feature importance from SHAP values
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # Create a dataframe for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_importance
    }).sort_values(by='Importance', ascending=False)
    
    top_features = feature_importance_df['Feature'].tolist()
    
    # Store feature rankings without prediction accuracy
    method_name = f"RF SHAP ({dataset_name})"
    feature_rankings[method_name] = top_features
    
    add_result(method_name, dataset_name, None, None, top_features)
    
    return feature_importance_df

# 7. SHAP for XGBoost - Updated to remove prediction accuracy
def run_xgb_shap(X_train, X_test, y_train, y_test, feature_names, xgb_model, dataset_name="Full"):
    # Create explainer
    explainer = shap.TreeExplainer(xgb_model)
    
    # Calculate SHAP values on a sample of training data (for computational efficiency)
    sample_size = min(100, X_train.shape[0])
    X_sample = X_train[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    
    # Calculate feature importance from SHAP values
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # Create a dataframe for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_importance
    }).sort_values(by='Importance', ascending=False)
    
    top_features = feature_importance_df['Feature'].tolist()
    
    # Store feature rankings without prediction accuracy
    method_name = f"XGB SHAP ({dataset_name})"
    feature_rankings[method_name] = top_features
    
    add_result(method_name, dataset_name, None, None, top_features)
    
    return feature_importance_df

# Convert data to numpy arrays
X_train_np = X_train.values
X_test_np = X_test.values
feature_names = X.columns.tolist()

print(f"Number of features: {len(feature_names)}")

# Run all methods on full dataset
print("Running methods on full dataset...")
rf_importance, rf_model = run_random_forest(X_train_np, X_test_np, y_train, y_test, feature_names)
xgb_importance, xgb_model = run_xgboost(X_train_np, X_test_np, y_train, y_test, feature_names)
fa_importance, fa_model = run_feature_agglomeration(X_train_np, X_test_np, y_train, y_test, feature_names)
hvgs_importance, hvgs_model = run_hvgs(X_train_np, X_test_np, y_train, y_test, feature_names)
spearman_importance, _ = run_spearman(X_train_np, X_test_np, y_train, y_test, feature_names)
rf_shap_importance = run_rf_shap(X_train_np, X_test_np, y_train, y_test, feature_names, rf_model)
xgb_shap_importance = run_xgb_shap(X_train_np, X_test_np, y_train, y_test, feature_names, xgb_model)

# Create reduced datasets by removing the top feature from each method
print("\nCreating reduced datasets...")

# For Random Forest
top_rf_feature = rf_importance['Feature'].iloc[0]
reduced_feature_names_rf = [f for f in feature_names if f != top_rf_feature]
reduced_idx_rf = [feature_names.index(f) for f in reduced_feature_names_rf]
X_train_rf_reduced = X_train_np[:, reduced_idx_rf]
X_test_rf_reduced = X_test_np[:, reduced_idx_rf]
print(f"RF: Removed top feature '{top_rf_feature}'")

# For XGBoost
top_xgb_feature = xgb_importance['Feature'].iloc[0]
reduced_feature_names_xgb = [f for f in feature_names if f != top_xgb_feature]
reduced_idx_xgb = [feature_names.index(f) for f in reduced_feature_names_xgb]
X_train_xgb_reduced = X_train_np[:, reduced_idx_xgb]
X_test_xgb_reduced = X_test_np[:, reduced_idx_xgb]
print(f"XGB: Removed top feature '{top_xgb_feature}'")

# For Feature Agglomeration
top_fa_feature = fa_importance['Feature'].iloc[0]
reduced_feature_names_fa = [f for f in feature_names if f != top_fa_feature]
reduced_idx_fa = [feature_names.index(f) for f in reduced_feature_names_fa]
X_train_fa_reduced = X_train_np[:, reduced_idx_fa]
X_test_fa_reduced = X_test_np[:, reduced_idx_fa]
print(f"FA: Removed top feature '{top_fa_feature}'")

# For HVGS
top_hvgs_feature = hvgs_importance['Feature'].iloc[0]
reduced_feature_names_hvgs = [f for f in feature_names if f != top_hvgs_feature]
reduced_idx_hvgs = [feature_names.index(f) for f in reduced_feature_names_hvgs]
X_train_hvgs_reduced = X_train_np[:, reduced_idx_hvgs]
X_test_hvgs_reduced = X_test_np[:, reduced_idx_hvgs]
print(f"HVGS: Removed top feature '{top_hvgs_feature}'")

# For Spearman
top_spearman_feature = spearman_importance['Feature'].iloc[0]
reduced_feature_names_spearman = [f for f in feature_names if f != top_spearman_feature]
reduced_idx_spearman = [feature_names.index(f) for f in reduced_feature_names_spearman]
X_train_spearman_reduced = X_train_np[:, reduced_idx_spearman]
X_test_spearman_reduced = X_test_np[:, reduced_idx_spearman]
print(f"Spearman: Removed top feature '{top_spearman_feature}'")

# For RF SHAP
top_rf_shap_feature = rf_shap_importance['Feature'].iloc[0]
reduced_feature_names_rf_shap = [f for f in feature_names if f != top_rf_shap_feature]
reduced_idx_rf_shap = [feature_names.index(f) for f in reduced_feature_names_rf_shap]
X_train_rf_shap_reduced = X_train_np[:, reduced_idx_rf_shap]
X_test_rf_shap_reduced = X_test_np[:, reduced_idx_rf_shap]
print(f"RF SHAP: Removed top feature '{top_rf_shap_feature}'")

# For XGB SHAP
top_xgb_shap_feature = xgb_shap_importance['Feature'].iloc[0]
reduced_feature_names_xgb_shap = [f for f in feature_names if f != top_xgb_shap_feature]
reduced_idx_xgb_shap = [feature_names.index(f) for f in reduced_feature_names_xgb_shap]
X_train_xgb_shap_reduced = X_train_np[:, reduced_idx_xgb_shap]
X_test_xgb_shap_reduced = X_test_np[:, reduced_idx_xgb_shap]
print(f"XGB SHAP: Removed top feature '{top_xgb_shap_feature}'")

# Run all methods on reduced datasets
print("\nRunning methods on reduced datasets...")

# RF
rf_reduced_importance, rf_reduced_model = run_random_forest(
    X_train_rf_reduced, X_test_rf_reduced, y_train, y_test, 
    reduced_feature_names_rf, "Reduced (RF)"
)

# XGB
xgb_reduced_importance, xgb_reduced_model = run_xgboost(
    X_train_xgb_reduced, X_test_xgb_reduced, y_train, y_test, 
    reduced_feature_names_xgb, "Reduced (XGB)"
)

# FA
fa_reduced_importance, _ = run_feature_agglomeration(
    X_train_fa_reduced, X_test_fa_reduced, y_train, y_test, 
    reduced_feature_names_fa, "Reduced (FA)"
)

# HVGS
hvgs_reduced_importance, _ = run_hvgs(
    X_train_hvgs_reduced, X_test_hvgs_reduced, y_train, y_test, 
    reduced_feature_names_hvgs, "Reduced (HVGS)"
)

# Spearman
spearman_reduced_importance, _ = run_spearman(
    X_train_spearman_reduced, X_test_spearman_reduced, y_train, y_test, 
    reduced_feature_names_spearman, "Reduced (Spearman)"
)

# RF SHAP (using reduced RF model) - No accuracy metrics
rf_shap_reduced_importance = run_rf_shap(
    X_train_rf_shap_reduced, X_test_rf_shap_reduced, y_train, y_test, 
    reduced_feature_names_rf_shap, rf_reduced_model, "Reduced (RF SHAP)"
)

# XGB SHAP (using reduced XGB model) - No accuracy metrics
xgb_shap_reduced_importance = run_xgb_shap(
    X_train_xgb_shap_reduced, X_test_xgb_shap_reduced, y_train, y_test, 
    reduced_feature_names_xgb_shap, xgb_reduced_model, "Reduced (XGB SHAP)"
)

# Create results dataframe for performance metrics and feature rankings
results_df = pd.DataFrame(results)

# Reorder the results to group by algorithm (full then reduced)
methods_order = [
    "Random Forest (Full)", "Random Forest (Reduced (RF))",
    "XGBoost (Full)", "XGBoost (Reduced (XGB))",
    "Feature Agglomeration (Full)", "Feature Agglomeration (Reduced (FA))",
    "HVGS (Full)", "HVGS (Reduced (HVGS))",
    "Spearman (Full)", "Spearman (Reduced (Spearman))",
    "RF SHAP (Full)", "RF SHAP (Reduced (RF SHAP))",
    "XGB SHAP (Full)", "XGB SHAP (Reduced (XGB SHAP))"
]

# Create a mapping for sorting
method_order_map = {method: i for i, method in enumerate(methods_order)}

# Use the mapping to sort the DataFrame
results_df['sort_order'] = results_df['method'].map(lambda x: method_order_map.get(x, 999))
results_df = results_df.sort_values('sort_order').drop('sort_order', axis=1)

# Export performance results and feature rankings to CSV
results_df.to_csv('feature_importance_results.csv', index=False)
print("\nResults saved to 'feature_importance_results.csv'")

# Print a preview of the results
print("\nFeature Importance Rankings and Prediction Accuracy Results:")
print(results_df[['method', 'dataset', 'train_r2', 'test_r2']].to_string(index=False))
print("\n[Full feature rankings included in the CSV output]")

# Create a summary of removed features
removed_features = {
    "Method": ["Random Forest", "XGBoost", "Feature Agglomeration", "HVGS", "Spearman", "RF SHAP", "XGB SHAP"],
    "Removed Feature": [top_rf_feature, top_xgb_feature, top_fa_feature, top_hvgs_feature, 
                        top_spearman_feature, top_rf_shap_feature, top_xgb_shap_feature]
}
removed_df = pd.DataFrame(removed_features)
removed_df.to_csv('removed_features.csv', index=False)
print("\nRemoved features summary saved to 'removed_features.csv'")
