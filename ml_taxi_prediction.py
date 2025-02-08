import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report

# Ensure pyarrow is installed
try:
    import pyarrow
except ImportError:
    raise ImportError("Missing optional dependency 'pyarrow'. Install it using: pip install pyarrow")

# Define dtypes explicitly
dtype_dict = {
    'RatecodeID': 'object',
    'airport_fee': 'float64',
    'congestion_surcharge': 'float64',
    'improvement_surcharge': 'float64',
    'passenger_count': 'float64',
    'tolls_amount': 'float64'
}

# Load dataset with specified dtypes
print("Loading dataset with specified dtypes...")
df = dd.read_csv("yellow_tripdata_2023-01.csv", parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"], 
                 dtype=dtype_dict, na_values=["\\N"], engine="pyarrow")

print("Dataset successfully loaded!")

# Select relevant columns
selected_columns = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "total_amount", "fare_amount"]
df = df[selected_columns]

# Convert to Pandas
df = df.compute()

# Handle missing values
df.dropna(inplace=True)

# Feature Engineering: Create trip duration
df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()

# Remove negative durations
df = df[df["trip_duration"] > 0]

# Show basic statistics
print(df[["trip_distance", "trip_duration", "total_amount"]].describe())

# Regression: Predict Total Fare
X = df[["trip_distance", "trip_duration"]]
y = df["total_amount"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predict and Evaluate Regression Model
y_pred = reg_model.predict(X_test)
print("Regression Model Evaluation:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")

# Classification: Predict High vs Low Fare
fare_threshold = df["total_amount"].median()
df["high_fare"] = (df["total_amount"] > fare_threshold).astype(int)

X_class = df[["trip_distance", "trip_duration"]]
y_class = df["high_fare"]

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_class, y_train_class)

# Predict and Evaluate Classification Model
y_pred_class = clf.predict(X_test_class)
print("Classification Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test_class, y_pred_class)}")
print(classification_report(y_test_class, y_pred_class))

# Visualization: Regression Predictions
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Regression Model: Actual vs Predicted Fares")
plt.show()

# Visualization: Feature Importance
feature_importances = clf.feature_importances_
sns.barplot(x=["Trip Distance", "Trip Duration"], y=feature_importances)
plt.title("Feature Importance in Classification Model")
plt.show()
