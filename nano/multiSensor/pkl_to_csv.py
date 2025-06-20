import joblib

# Load the DataFrame
df = joblib.load("X_filtered.pkl")

# Save to CSV
df.to_csv("X_filtered.csv", index=False)

print("✅ Successfully saved as CSV!")
