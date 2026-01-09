import joblib
import numpy as np

# Load the preprocessor
preprocessor = joblib.load('models/burnout_v8.pkl')

print("=" * 60)
print("PREPROCESSOR ANALYSIS")
print("=" * 60)

print(f"\n1. COMPONENTS AVAILABLE:")
for key in preprocessor.keys():
    print(f"   • {key}")

print(f"\n2. FEATURE INFORMATION:")
print(f"   Total features: {len(preprocessor['feature_names'])}")
print(f"   First 5 features:")
for i, name in enumerate(preprocessor['feature_names'][:5]):
    print(f"     {i+1}. {name}")

print(f"\n3. IMPUTER STATISTICS:")
print(f"   Strategy: {preprocessor['imputer'].strategy}")
print(f"   First 5 median values:")
for i, median in enumerate(preprocessor['imputer'].statistics_[:5]):
    print(f"     {i+1}. {median:.3f}")

print(f"\n4. SCALER PARAMETERS:")
print(f"   Samples seen: {preprocessor['scaler'].n_samples_seen_}")
print(f"   First 5 means: {preprocessor['scaler'].mean_[:5].round(3)}")
print(f"   First 5 std devs: {preprocessor['scaler'].scale_[:5].round(3)}")

print(f"\n5. CATEGORICAL HANDLING:")
print(f"   Categorical columns: {preprocessor['categorical_columns']}")
print(f"   Label encoders: {len(preprocessor['label_encoders'])}")

print("\n" + "=" * 60)