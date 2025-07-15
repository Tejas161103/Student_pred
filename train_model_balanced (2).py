
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# Load dataset
df = pd.read_csv("student-mat.csv", sep=";")
df.columns = df.columns.str.strip()  # clean column names

print("🔍 First 5 rows of the dataset:")
print(df.head())

# Create binary 'pass' column
df["pass"] = (df["G3"] >= 10).astype(int)

print("\n📊 G3 value counts:")
print(df["G3"].value_counts())

print("\n✅ Total records in dataset:", len(df))
print("✅ Class distribution:\n", df["pass"].value_counts())

# ✅ Balance dataset safely
df_pass = df[df["pass"] == 1]
df_fail = df[df["pass"] == 0]

if len(df_pass) == 0 or len(df_fail) == 0:
    raise ValueError("❌ Dataset must contain both pass and fail records.")

n_samples = min(len(df_pass), len(df_fail))

df_pass_sampled = df_pass.sample(n=n_samples, random_state=42)
df_fail_sampled = df_fail.sample(n=n_samples, random_state=42)

df_balanced = pd.concat([df_pass_sampled, df_fail_sampled])

# Feature and target
X = df_balanced[["studytime", "failures", "absences"]]
y = df_balanced["pass"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
with open("model/student_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")

# 🧪 Show predictions for sanity check
print("\n🧪 Sample Predictions:")
for studytime in range(1, 5):
    for failures in range(0, 3):
        for absences in [0, 5, 10]:
            pred = model.predict([[studytime, failures, absences]])[0]
            print(f"→ studytime={studytime}, failures={failures}, absences={absences} → {'Pass 👍' if pred == 1 else 'Fail 👎'}")
