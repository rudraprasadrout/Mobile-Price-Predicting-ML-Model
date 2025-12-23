import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# 1. Load the Model and Data
print("Loading model and data...")
try:
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")   
    
    # We use input.csv because it contains the separate test data with actual prices
    df_test = pd.read_csv("input.csv") 
except FileNotFoundError as e:
    print(f"Error: Required file not found. {e}")
    exit()

# 2. Separate Features and Actual Answers
# We need to make sure we don't accidentally feed the 'Price' into the input
if "Price" not in df_test.columns:
    print("Error: 'Price' column missing in input.csv. Cannot calculate accuracy.")
    exit()

y_actual = df_test["Price"]              # The real answers
X_test = df_test.drop("Price", axis=1)   # The questions (features)

# 3. Predict
# Transform the data using the loaded pipeline
print("Running predictions...")
X_test_prepared = pipeline.transform(X_test)
predictions = model.predict(X_test_prepared)

# 4. Calculate Accuracy Metrics
mse = mean_squared_error(y_actual, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, predictions)
mape = mean_absolute_percentage_error(y_actual, predictions)
accuracy_percentage = 100 - (mape * 100)

# 5. Print the Report
print("\n" + "="*40)
print("      MODEL PERFORMANCE REPORT      ")
print("="*40)
print(f"RMSE (Avg Error)    : {rmse:.2f}")
print(f"R2 Score (Fit)      : {r2:.4f}  (1.0 is perfect)")
print(f"MAPE (Error %)      : {mape*100:.2f}%")
print("-" * 40)
print(f"FINAL ACCURACY      : {accuracy_percentage:.2f}%")
print("="*40)

# 6. Generate a Graph (Actual vs Predicted)
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Scatter plot of points
sns.scatterplot(x=y_actual, y=predictions, alpha=0.6, color='blue', edgecolor='k', s=80)

# Draw a red diagonal line (The "Perfect Prediction" line)
min_val = min(min(y_actual), min(predictions))
max_val = max(max(y_actual), max(predictions))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit Line')

plt.title(f"Actual vs Predicted Price (Accuracy: {accuracy_percentage:.1f}%)", fontsize=14)
plt.xlabel("Actual Price", fontsize=12)
plt.ylabel("Predicted Price", fontsize=12)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('accuracy_plot.png')
print("\nGraph saved as 'accuracy_plot.png'. Open it to see visual performance!")
plt.show()