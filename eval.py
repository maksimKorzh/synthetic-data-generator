import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# Download data
for filename in ['real.csv', 'synthetic_raw.csv']:
    with open(filename, 'w') as f:
        data = requests.get('https://github.com/maksimKorzh/synthetic-data-generator/releases/download/0.1/synthetic_raw.csv').text
        features = 'Suburb,Address,Rooms,Type,Price,Method,SellerG,Date,Distance,Postcode,Bedroom2,Bathroom,Car,Landsize,BuildingArea,YearBuilt,CouncilArea,Lattitude,Longtitude,Regionname,Propertycount\n'
        f.write(features + data)

# Load synthetic data
df = pd.read_csv("synthetic_raw.csv", on_bad_lines='warn', low_memory=False)

# Ensure expected columns exist
expected_columns = ["Price", "Rooms", "Landsize", "Distance", "Type", "Suburb", "Regionname"]
missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns in CSV: {missing_cols}")

# ----------- NUMERIC CLEANING ----------- #

def safe_convert(col, to_type=float):
    return pd.to_numeric(df[col], errors='coerce')

df['Price'] = safe_convert("Price")
df['Rooms'] = safe_convert("Rooms")
df['Landsize'] = safe_convert("Landsize")
df['Distance'] = safe_convert("Distance")

# Drop negative or extreme values
df['Rooms'] = df['Rooms'].clip(lower=1, upper=10).round()
df['Distance'] = df['Distance'].clip(lower=0, upper=50)
df['Landsize'] = df['Landsize'].clip(lower=0, upper=5000)
df['Price'] = df['Price'].clip(lower=100000, upper=5000000)

# Drop rows with any NA in required columns
df.dropna(subset=["Price", "Rooms", "Distance", "Landsize", "Type", "Suburb", "Regionname"], inplace=True)

# ----------- CATEGORICAL CLEANING ----------- #

# Keep only known valid types
valid_types = {"h", "u", "t"}
df['Type'] = df['Type'].str.strip().str.lower()
df = df[df['Type'].isin(valid_types)]

# Optional: simplify suburb/region to match real data if model hallucinated
# Drop or normalize uncommon ones
suburb_counts = df['Suburb'].value_counts()
valid_suburbs = suburb_counts[suburb_counts > 5].index
df = df[df['Suburb'].isin(valid_suburbs)]

region_counts = df['Regionname'].value_counts()
valid_regions = region_counts[region_counts > 5].index
df = df[df['Regionname'].isin(valid_regions)]

# ----------- FINAL CHECKS ----------- #

print("Cleaned synthetic data summary:")
print(df.describe(include='all'))

# Save cleaned version
df.to_csv("synthetic.csv", index=False)
print("Saved cleaned data to synthetic_cleaned.csv")

# --- Load Data ---
df_synth = pd.read_csv("synthetic.csv", on_bad_lines='skip', low_memory=False)
df_real = pd.read_csv("real.csv", on_bad_lines='skip', low_memory=False)

# Clean price column
df_synth['Price'] = pd.to_numeric(df_synth['Price'], errors='coerce')
df_synth = df_synth.dropna(subset=['Price'])

df_real['Price'] = pd.to_numeric(df_real['Price'], errors='coerce')
df_real = df_real.dropna(subset=['Price'])

# Select features & target
features = ['Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car',
            'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
target = 'Price'

# Subset real dataset to same as synth
df_real = df_real.iloc[:len(df_synth)]

X_synth = df_synth[features].copy()
y_synth = df_synth[target].copy()

X_real = df_real[features].copy()
y_real = df_real[target].copy()

# Identify numeric and categorical features
numeric_features = [col for col in features if col != 'Type']
categorical_features = ['Type']

# Coerce numeric columns to numeric dtype, invalid parsing -> NaN
for col in numeric_features:
    X_synth[col] = pd.to_numeric(X_synth[col], errors='coerce')
    X_real[col] = pd.to_numeric(X_real[col], errors='coerce')

# Drop rows with NaNs in numeric features or target
df_synth_clean = pd.concat([X_synth, y_synth], axis=1).dropna(subset=numeric_features + [target])
df_real_clean = pd.concat([X_real, y_real], axis=1).dropna(subset=numeric_features + [target])

X_synth = df_synth_clean[features]
y_synth = df_synth_clean[target]

X_real = df_real_clean[features]
y_real = df_real_clean[target]

# --- Preprocessing pipelines ---
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Helper function to train and evaluate
def train_evaluate(model, model_name, X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} | Train size: {len(X_train):,} | Test size: {len(X_test):,} | MAE: {mae:,.2f} | R2: {r2:.3f}")
    return pipe, y_pred, mae, r2

# Define models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

# Prepare subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

scenarios = [
    # (model, model_name, X_train, y_train, X_test, y_test, subplot_index, title)
    (rf_model, "Random Forest", X_real, y_real, X_real, y_real, 0, "RF: Train Real, Test Real"),
    (rf_model, "Random Forest", X_synth, y_synth, X_synth, y_synth, 1, "RF: Train Synth, Test Synth"),
    (rf_model, "Random Forest", X_synth, y_synth, X_real, y_real, 2, "RF: Train Synth, Test Real"),
    (xgb_model, "XGBoost", X_real, y_real, X_real, y_real, 3, "XGB: Train Real, Test Real"),
    (xgb_model, "XGBoost", X_synth, y_synth, X_synth, y_synth, 4, "XGB: Train Synth, Test Synth"),
    (xgb_model, "XGBoost", X_synth, y_synth, X_real, y_real, 5, "XGB: Train Synth, Test Real"),
]

for model, name, X_tr, y_tr, X_te, y_te, idx, title in scenarios:
    pipe, y_pred, mae, r2 = train_evaluate(model, name, X_tr, y_tr, X_te, y_te)
    ax = axes[idx]
    ax.scatter(y_te, y_pred, alpha=0.4)
    ax.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title(title)
    title_with_mae = f"{title}\nMAE: {mae:,.0f}"
    ax.set_title(title_with_mae)
    ax.grid(True)

plt.suptitle("House Price Prediction: Real vs Synthetic Data Comparison", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
