import pandas as pd
import numpy as np
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import linregress, mannwhitneyu

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import hdbscan
from hdbscan import validity

# Folders
FIG_DIR = Path("figures")

(FIG_DIR / "temperature").mkdir(parents=True, exist_ok=True)
(FIG_DIR / "correlation").mkdir(parents=True, exist_ok=True)
(FIG_DIR / "pca").mkdir(parents=True, exist_ok=True)
(FIG_DIR / "clustering").mkdir(parents=True, exist_ok=True)
(FIG_DIR / "risk").mkdir(parents=True, exist_ok=True)
(FIG_DIR / "tables").mkdir(parents=True, exist_ok=True)


# Data
dc_temp = pd.read_csv("temperatures_2006_2025datacenters.csv")
city_temp = pd.read_csv("temperatures_2006_2025citycenters.csv")

dc_water = pd.read_csv("dc_water_stress_results.csv")
city_water = pd.read_csv("cities_water_stress_results.csv")


def build_yearly_climate_features(df, location_type):
    group_cols = ["country", "latitude", "longitude"]

    df = df.sort_values(group_cols + ["year"]).copy()

    df["temp_yoy_change"] = df.groupby(group_cols)["avg_temp_c"].diff()

    first_temps = df.groupby(group_cols)["avg_temp_c"].transform("first")
    df["temp_cumulative_change"] = df["avg_temp_c"] - first_temps

    df["location_type"] = location_type

    return df


# Process both datasets to maintain year-by-year rows
dc_features = build_yearly_climate_features(dc_temp, "DataCenter")
city_features = build_yearly_climate_features(city_temp, "City")


# Merging
spatial_keys = ["country", "latitude", "longitude"]

dc_df = dc_features.merge(dc_water, on=spatial_keys, how="left")
city_df = city_features.merge(city_water, on=spatial_keys, how="left")

master_df = pd.concat([dc_df, city_df], ignore_index=True)

master_df["target"] = (master_df["location_type"] == "DataCenter").astype(int)

features = ["year", "temp_yoy_change", "temp_cumulative_change", "bws_raw", "drr_raw"]

summary = master_df.groupby("location_type")[features].describe()

print("\nSUMMARY STATISTICS")
print(summary)

summary.to_csv(FIG_DIR / "tables" / "summary_statistics.csv")

stats_results = []

print("\nMANN-WHITNEY TESTS")

for col in features:
    dc = dc_df[col].dropna()
    city = city_df[col].dropna()

    stat, p = mannwhitneyu(dc, city)

    print(f"{col}: p={p:.6f}")

    stats_results.append({"feature": col, "statistic": stat, "p_value": p})

# Save statistical test outputs
pd.DataFrame(stats_results).to_csv(
    FIG_DIR / "tables" / "mannwhitney_results.csv", index=False
)


for feature in features:
    plt.figure(figsize=(8, 6))

    sns.boxplot(data=master_df, x="location_type", y=feature)

    plt.title(f"{feature}: Data Centers vs Cities (2006-2025)")
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "temperature" / f"{feature}_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


corr_cols = ["temp_yoy_change", "temp_cumulative_change", "bws_raw", "drr_raw"]

corr = master_df[corr_cols].corr()

corr.to_csv(FIG_DIR / "tables" / "correlation_matrix.csv")

plt.figure(figsize=(8, 6))

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)

plt.title("Dynamic Climate & Water Correlation Matrix")
plt.tight_layout()

plt.savefig(
    FIG_DIR / "correlation" / "correlation_heatmap.png", dpi=300, bbox_inches="tight"
)
plt.close()

cluster_features = ["temp_yoy_change", "temp_cumulative_change", "bws_raw", "drr_raw"]

X = master_df[cluster_features].copy()

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)

master_df["PC1"] = pca_coords[:, 0]
master_df["PC2"] = pca_coords[:, 1]

# Plot the results
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=master_df,
    x="PC1",
    y="PC2",
    hue="location_type",
    alpha=0.4,
    edgecolor=None,
)

plt.title("PCA: Data Centers vs Cities (Dynamic Yearly Risks)")
plt.tight_layout()
plt.savefig(FIG_DIR / "pca" / "pca_location_types.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")


def run_hdbscan_analysis(master_df, features, title_suffix=""):
    group_cols = ["country", "latitude", "longitude", "location_type"]

    location_profiles = master_df.groupby(group_cols)[features].mean().reset_index()

    X = location_profiles[features].copy()
    X = X.fillna(X.median())
    X_scaled = StandardScaler().fit_transform(X)

    best_score = -np.inf
    best_model = None
    best_params = None

    for min_cluster_size in range(4, 25):
        for min_samples in range(4, 25):
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                core_dist_n_jobs=-1,  # Uses all CPU cores to speed things up
            )

            labels = model.fit_predict(X_scaled)
            n_clusters = len(set(labels) - {-1})

            if n_clusters < 2:
                continue

            try:
                # DBCV validation check
                score = hdbscan.validity.validity_index(X_scaled, labels)
            except:
                continue

            if score > best_score:
                best_score = score
                best_model = model
                best_params = (min_cluster_size, min_samples)

    if best_model is None:
        raise ValueError(
            "HDBSCAN could not discover valid multi-cluster arrangements. Try lowering parameters below 4."
        )

    location_profiles["cluster"] = best_model.labels_
    location_profiles["outlier_score"] = best_model.outlier_scores_
    location_profiles["cluster_probability"] = best_model.probabilities_

    print(f"\n===== {title_suffix} =====")
    print("Best parameters (min_size, min_samples):", best_params)
    print("Best DBCV score:", round(best_score, 4))
    print(f"Clusters discovered: {len(set(best_model.labels_) - {-1})}")

    map_cols = group_cols + ["cluster", "outlier_score", "cluster_probability"]
    updated_master_df = master_df.merge(
        location_profiles[map_cols], on=group_cols, how="left"
    )

    return (updated_master_df, best_model, best_score, best_params)


(
    combined_clusters,
    combined_model,
    combined_score,
    combined_params,
) = run_hdbscan_analysis(master_df, cluster_features, "COMBINED")

dc_clusters, dc_model, dc_score, dc_params = run_hdbscan_analysis(
    dc_df, cluster_features, "DATA CENTERS"
)

city_clusters, city_model, city_score, city_params = run_hdbscan_analysis(
    city_df, cluster_features, "CITY CENTERS"
)


combined_clusters.to_csv(FIG_DIR / "clustering" / "combined_clusters.csv", index=False)
dc_clusters.to_csv(FIG_DIR / "clustering" / "datacenter_clusters.csv", index=False)
city_clusters.to_csv(FIG_DIR / "clustering" / "city_clusters.csv", index=False)


combined_summary = combined_clusters.groupby("cluster")[cluster_features].mean()
dc_summary = dc_clusters.groupby("cluster")[cluster_features].mean()
city_summary = city_clusters.groupby("cluster")[cluster_features].mean()

combined_summary.to_csv(FIG_DIR / "clustering" / "combined_cluster_profiles.csv")
dc_summary.to_csv(FIG_DIR / "clustering" / "datacenter_cluster_profiles.csv")
city_summary.to_csv(FIG_DIR / "clustering" / "city_cluster_profiles.csv")

print("\nCOMBINED CLUSTER PROFILES")
print(combined_summary)

cluster_mix = pd.crosstab(
    combined_clusters["cluster"], combined_clusters["location_type"]
)

print("\nCLUSTER MIX")
print(cluster_mix)

cluster_mix.to_csv(FIG_DIR / "clustering" / "combined_cluster_mix.csv")

combined_clusters["PC1"] = master_df["PC1"].values
combined_clusters["PC2"] = master_df["PC2"].values

plt.figure(figsize=(10, 8))

sns.scatterplot(
    data=combined_clusters,
    x="PC1",
    y="PC2",
    hue="cluster",
    style="location_type",
    palette="tab20",
    alpha=0.5,
    edgecolor=None,
)

plt.title("Environmental Risk Clusters (Timeline Spatial Distribution)")
plt.tight_layout()

plt.savefig(FIG_DIR / "pca" / "pca_clusters.png", dpi=300, bbox_inches="tight")
plt.close()

from sklearn.model_selection import GroupKFold

X_rf = master_df[cluster_features].copy().fillna(master_df[cluster_features].median())
y_rf = master_df["target"]

spatial_groups = (
    master_df["country"]
    + "_"
    + master_df["latitude"].astype(str)
    + "_"
    + master_df["longitude"].astype(str)
)

gkf = GroupKFold(n_splits=5)
train_idx, test_idx = next(gkf.split(X_rf, y_rf, groups=spatial_groups))

X_train_raw, X_test = X_rf.iloc[train_idx], X_rf.iloc[test_idx]
y_train_raw, y_test = y_rf.iloc[train_idx], y_rf.iloc[test_idx]

train_df = X_train_raw.copy()
train_df["target"] = y_train_raw

df_class_0 = train_df[train_df["target"] == 0]
df_class_1 = train_df[train_df["target"] == 1]

minority_size = len(df_class_0)

df_class_1_downsampled = df_class_1.sample(n=minority_size, random_state=42)

balanced_train_df = pd.concat([df_class_0, df_class_1_downsampled], ignore_index=True)

balanced_train_df = balanced_train_df.sample(frac=1, random_state=42).reset_index(
    drop=True
)

X_train = balanced_train_df[cluster_features]
y_train = balanced_train_df["target"]

rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("\nRANDOM FOREST RESULTS (DOWNSAMPLED & GROUP-SAFE)")
print(classification_report(y_test, pred))

importance = pd.DataFrame(
    {"feature": cluster_features, "importance": rf.feature_importances_}
)
importance.sort_values("importance", ascending=False, inplace=True)

print("\nFEATURE IMPORTANCE")
print(importance)

importance.to_csv(FIG_DIR / "risk" / "feature_importance.csv", index=False)

plt.figure(figsize=(8, 6))
importance_plot = importance.sort_values("importance", ascending=True)
plt.barh(importance_plot["feature"], importance_plot["importance"])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance (Balanced Data)")
plt.tight_layout()
plt.savefig(
    FIG_DIR / "risk" / "rf_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)


hdbscan_summary = pd.DataFrame(
    {
        "dataset": ["combined", "datacenters", "cities"],
        "dbcv_score": [combined_score, dc_score, city_score],
        "min_cluster_size": [combined_params[0], dc_params[0], city_params[0]],
        "min_samples": [combined_params[1], dc_params[1], city_params[1]],
    }
)

hdbscan_summary.to_csv(FIG_DIR / "clustering" / "hdbscan_summary.csv", index=False)

print("\nHDBSCAN SUMMARY")
print(hdbscan_summary)

print("\nAnalysis complete.")
print(f"Results saved to: {FIG_DIR.resolve()}")

water_check_cols = ["country", "bws_raw"]
if "city" in master_df.columns:
    water_check_cols.append("city")
if "bws_cat" in master_df.columns:
    water_check_cols.append("bws_cat")
if "bws_label" in master_df.columns:
    water_check_cols.append("bws_label")

print("\nLOCATIONS EXCEEDING WATER STRESS THRESHOLD:")
print(master_df[master_df["bws_raw"] > 1][water_check_cols])
