import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

df = pd.read_csv("dataset.csv")

size_columns = ["SIZE_X", "SIZE_Y", "DEFECT_AREA"]
cleaned_parts = []
for class_id, df_class in df.groupby("Class"):
    df_class = df_class.copy()
    for col in size_columns:
        if df_class[col].count() < 2:
            continue
        q1 = df_class[col].quantile(0.25)
        q3 = df_class[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        upper = q3 + 1.5 * iqr
        df_class = df_class[df_class[col] <= upper]
    cleaned_parts.append(df_class)

df_cleaned = pd.concat(cleaned_parts)

df_cleaned["SNR_OFFSET_GL"] = df_cleaned["MDAT_OFFSET"] / (df_cleaned["MDAT_GL"] + 1e-6)
df_cleaned["SNR_INTENSITY_NOISE"] = df_cleaned["INTENSITY"] / (df_cleaned["PATCHNOISE"] + 1e-6)
df_cleaned["ASPECT_RATIO"] = df_cleaned["SIZE_X"] / (df_cleaned["SIZE_Y"] + 1e-6)
df_cleaned["ASPECT_RATIO"].replace([np.inf, -np.inf], np.nan, inplace=True)
df_cleaned["DENSITY_SIGNAL"] = df_cleaned["INTENSITY"] / (df_cleaned["DEFECT_AREA"] + 1e-6)
df_cleaned["DENSITY_SIGNAL"].replace([np.inf, -np.inf], np.nan, inplace=True)

process_steps = ["PC", "RMG", "CBCMP"]
numerical_features = [
    "ENERGY_PARAM",
    "MDAT_OFFSET",
    "RELATIVEMAGNITUDE",
    "PATCHDEFECTSIGNAL",
    "INTENSITY",
    "POLARITY",
    "MDAT_GL",
    "MDAT_NOISE",
    "PATCHNOISE",
    "SIZE_X",
    "SIZE_Y",
    "DEFECT_AREA",
    "SIZE_D",
    "RADIUS",
    "ANGLE",
    "ALIGNRATIO",
    "SPOTLIKENESS",
    "ACTIVERATIO",
]

df_cleaned["KMeans_Cluster"] = pd.NA

for step in process_steps:
    step_real = df_cleaned[(df_cleaned["IS_DEFECT"] == "REAL") & (df_cleaned["Step_desc"] == step)].copy()
    step_real = step_real.dropna(subset=numerical_features)
    if step_real.empty:
        continue
    scaler = StandardScaler()
    scaled = scaler.fit_transform(step_real[numerical_features])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)
    df_cleaned.loc[step_real.index, "KMeans_Cluster"] = labels

df_cleaned["KMeans_Cluster"] = df_cleaned["KMeans_Cluster"].astype("Int64")

killer_cluster_mapping = {"PC": 1, "RMG": 1, "CBCMP": 0}
df_cleaned["is_killer_defect"] = False
for step, cluster_id in killer_cluster_mapping.items():
    mask = (
        (df_cleaned["IS_DEFECT"] == "REAL")
        & (df_cleaned["Step_desc"] == step)
        & (df_cleaned["KMeans_Cluster"] == cluster_id)
    )
    df_cleaned.loc[mask, "is_killer_defect"] = True

killer_counts = (
    df_cleaned[df_cleaned["is_killer_defect"]]
    .groupby("Lot Name")
    .size()
    .reset_index(name="Killer_Defect_Count")
)
total_real = (
    df_cleaned[df_cleaned["IS_DEFECT"] == "REAL"]
    .groupby("Lot Name")
    .size()
    .reset_index(name="Total_Count")
)
false_counts = (
    df_cleaned[df_cleaned["IS_DEFECT"] == "FALSE"]
    .groupby("Lot Name")
    .size()
    .reset_index(name="False_Defect_Count")
)
slot_counts = (
    df_cleaned.groupby("Lot Name")["Slot No"]
    .nunique()
    .reset_index(name="Slot_No_nunique")
)

df_lot = total_real.merge(killer_counts, on="Lot Name", how="left")
df_lot["Killer_Defect_Count"].fillna(0, inplace=True)
df_lot = df_lot.merge(false_counts, on="Lot Name", how="left")
df_lot["False_Defect_Count"].fillna(0, inplace=True)
df_lot = df_lot.merge(slot_counts, on="Lot Name", how="left")
df_lot["Slot_No_nunique"].fillna(1, inplace=True)
df_lot["Killer_Defect_Proportion"] = (
    df_lot["Killer_Defect_Count"] / df_lot["Total_Count"]
)

df_lot["Nuisance_Count"] = df_lot["Total_Count"] - df_lot["Killer_Defect_Count"]
df_lot["Killer_Defect_Count_per_slot"] = df_lot["Killer_Defect_Count"] / (
    df_lot["Slot_No_nunique"] + 1e-6
)
df_lot["Nuisance_Count_per_slot"] = df_lot["Nuisance_Count"] / (
    df_lot["Slot_No_nunique"] + 1e-6
)
df_lot["False_Defect_Count_per_slot"] = df_lot["False_Defect_Count"] / (
    df_lot["Slot_No_nunique"] + 1e-6
)

scaler_killer = MinMaxScaler()
scaler_nuisance = MinMaxScaler()
scaler_false = MinMaxScaler()

df_lot["Score_Killer"] = scaler_killer.fit_transform(
    df_lot[["Killer_Defect_Count_per_slot"]]
)
df_lot["Score_Nuisance"] = scaler_nuisance.fit_transform(
    df_lot[["Nuisance_Count_per_slot"]]
)
df_lot["Score_False"] = scaler_false.fit_transform(
    df_lot[["False_Defect_Count_per_slot"]]
)

w_killer, w_nuisance, w_false = 0.50, 0.30, 0.20
df_lot["Total_Risk_Score"] = (
    w_killer * df_lot["Score_Killer"]
    + w_nuisance * df_lot["Score_Nuisance"]
    + w_false * df_lot["Score_False"]
)

print("max", df_lot["Total_Risk_Score"].max())
print(df_lot.sort_values("Total_Risk_Score", ascending=False).head())

