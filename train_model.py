import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load datasets with correct separator and handle spaces in column names
eclipse_jdt = pd.read_csv("./data/Eclipse_ JDT_Core_single-version-ck-oo_bugs_only.csv", sep=';')
eclipse_pdt = pd.read_csv("./data/Eclipse_PDE_UI_single-version-ck-oo_bug_only.csv", sep=';')
equinox = pd.read_csv("./data/Equinox_Framework_single-version-ck-oo_bug_only.csv", sep=';')
lucene = pd.read_csv("./data/Lucene_single-version-ck-oo_bug_only.csv", sep=';')
mylyn = pd.read_csv("./data/Mylyn_single-version-ck-oo_bug_only.csv", sep=';')

# Clean column names
for df in [eclipse_jdt, eclipse_pdt, equinox, lucene, mylyn]:
    df.columns = df.columns.str.strip()

# Clean data
eclipse_pdt.dropna(axis=1, inplace=True)
equinox.dropna(axis=1, inplace=True)
lucene.dropna(axis=1, inplace=True)
mylyn.dropna(axis=1, inplace=True)

# Combine datasets
df = pd.concat([eclipse_jdt, eclipse_pdt, equinox, lucene, mylyn], ignore_index=True)

# Prepare features and target
feature_cols = [
    "cbo", "dit", "fanIn", "fanOut", "lcom", "noc", "numberOfAttributes", "numberOfAttributesInherited",
    "numberOfLinesOfCode", "numberOfMethods", "numberOfMethodsInherited", "numberOfPrivateAttributes",
    "numberOfPrivateMethods", "numberOfPublicAttributes", "numberOfPublicMethods", "rfc", "wmc"
]
X = df[feature_cols]
y = (df["bugs"] > 0).astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully!") 