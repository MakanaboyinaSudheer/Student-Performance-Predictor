import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1. DATA LOADING
print("Loading dataset...")
try:
    df = pd.read_csv("StudentPerformanceFactors.csv")
    
    # Rename columns to match what app.py expects and the rest of this script expects
    rename_map = {
        'Hours_Studied': 'study_hours',
        'Attendance': 'attendance',
        'Parental_Involvement': 'parental_participation',
        'Access_to_Resources': 'learning_resources',
        'Sleep_Hours': 'sleep_hours',
        'Motivation_Level': 'motivation_level',
        'Internet_Access': 'internet_availability',
        'Family_Income': 'family_background',
        'Teacher_Quality': 'teacher_quality',
        'School_Type': 'school_type',
        'Peer_Influence': 'peer_influence',
        'Exam_Score': 'exam_score'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Keep only the features that the web app provides, plus target
    expected_cols = [
        'study_hours', 'attendance', 'motivation_level', 'learning_resources',
        'parental_participation', 'teacher_quality', 'peer_influence', 'family_background',
        'internet_availability', 'sleep_hours', 'school_type', 'exam_score'
    ]
    df = df[[c for c in expected_cols if c in df.columns]]
    
except FileNotFoundError:
    print("Dataset not found. Please place 'StudentPerformanceFactors.csv' in the project folder.")
    print("Exiting...")
    exit(1)

# Display first few rows and info
print("\n--- FIRST 5 ROWS ---")
print(df.head())
print("\n--- DATASET INFO ---")
print(df.info())
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 2. DATA PREPROCESSING
print("\nPreprocessing dataset...")
# Handle missing values 
# Categorical filled with mode, Numeric filled with median
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])

# Separate features (X) and target (y)
X = df.drop('exam_score', axis=1)
y = df['exam_score']

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

# Define preprocessing for numeric columns (scale them)
numeric_transformer = StandardScaler()

# Define preprocessing for categorical features (encode them)
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Combine preprocessing steps into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. FEATURE ENGINEERING 
# (Grades and Risk Levels are applied dynamically inside Flask using exam_score)
def get_grade(score):
    if score >= 85: return 'A'
    elif score >= 70: return 'B'
    elif score >= 50: return 'C'
    else: return 'Fail'

def get_risk(score):
    if score >= 70: return 'Low'
    elif score >= 50: return 'Medium'
    else: return 'High'

# We map these to our dataframe for EDA purposes
df['Grade'] = df['exam_score'].apply(get_grade)
df['Risk_Level'] = df['exam_score'].apply(get_risk)

# 4. EXPLORATORY DATA ANALYSIS
print("Performing EDA (Saving plots as PNG)...")
# Heatmap (Only on numeric data to avoid errors)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Study hours vs exam score
if 'study_hours' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='study_hours', y='exam_score', data=df)
    plt.title("Study Hours vs Exam Score")
    plt.savefig("study_hours_vs_score.png")
    plt.close()

# Attendance vs exam score
if 'attendance' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='attendance', y='exam_score', data=df)
    plt.title("Attendance vs Exam Score")
    plt.savefig("attendance_vs_score.png")
    plt.close()

# Sleep vs performance
if 'sleep_hours' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='sleep_hours', y='exam_score', data=df)
    plt.title("Sleep Hours vs Exam Score")
    plt.savefig("sleep_vs_score.png")
    plt.close()

# 5. MODEL BUILDING & EVALUATION
print("\nTraining models...")
models = {
    "Linear Regression": Pipeline(steps=[('preprocessor', preprocessor),
                                          ('regressor', LinearRegression())]),
    "Decision Tree": Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', DecisionTreeRegressor(random_state=42))]),
    "Random Forest": Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
}

results = {}
best_model_name = None
best_r2 = -float('inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "R2": r2, "model": model}
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name

print("\n--- Model Evaluation ---")
for name, metrics in results.items():
    print(f"{name} -> RMSE: {metrics['RMSE']:.2f}, R² Score: {metrics['R2']:.2f}")

final_model = results[best_model_name]["model"]
print(f"\nSelected Best Model: {best_model_name} (R² = {best_r2:.2f})")

# 7. FEATURE IMPORTANCE / COEFFICIENTS
print(f"Plotting Feature Significance for {best_model_name}...")
try:
    fitted_preprocessor = final_model.named_steps['preprocessor']
    cat_encoder = fitted_preprocessor.named_transformers_['cat']
    cat_ftrs = list(cat_encoder.get_feature_names_out(categorical_features))
    all_ftrs = numeric_features + cat_ftrs

    regressor = final_model.named_steps['regressor']
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        title_str = "Top 10 Feature Importances"
    elif hasattr(regressor, 'coef_'):
        importances = np.abs(regressor.coef_)
        title_str = "Top 10 Feature Coefficients (Magnitude)"
    else:
        importances = None

    if importances is not None:
        importance_df = pd.DataFrame({'Feature': all_ftrs, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f"{title_str} ({best_model_name})")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
        print("Feature importance plot saved as feature_importance.png")
except Exception as e:
    print("Could not compute feature importance plot:", e)

# 8. SAVE MODEL
print("Saving the final Pipeline model as 'model.pkl'...")
with open("model.pkl", "wb") as f:
    pickle.dump(final_model, f)
print("Model created and saved successfully! Ready for Flask deployment.")

