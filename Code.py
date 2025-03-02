import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# Load the dataset
df = pd.read_csv("data.csv")

# Extract numeric values from 'Income_Percentage' column
if 'Income_Percentage' in df.columns:
    df['Income_Percentage'] = df['Income_Percentage'].astype(str)  # Convert to string first to handle non-numeric characters
    df['Income_Percentage'] = df['Income_Percentage'].str.extract(r'(\d+)')  # Extract numeric part
    df['Income_Percentage'] = pd.to_numeric(df['Income_Percentage'], errors='coerce')  # Convert to numeric
    df['Income_Percentage'].fillna(df['Income_Percentage'].mean(), inplace=True)  # Fill missing values with mean

# Check for missing values before preprocessing
print("Missing values before preprocessing:")
print(df.isnull().sum())

# 1. Simplify column names (ensure the list has 20 names)
df.columns = [
    'Timestamp', 'Age_Group', 'Gender', 'Style', 'Location', 'Clothing_Preference', 
    'Wardrobe_Essentials', 'Fashion_Inspiration', 'Shopping_Frequency', 
    'Shopping_Location', 'Shopping_Timing', 'Fashion_Emergency', 
    'Designer_Brands', 'Favorite_Store', 'Style_Change', 'Style_Age_Fit',
    'Society_Expectations', 'Spending', 'Income_Percentage', 'Fashion_Trend_Scale'
]

# 2. Drop irrelevant columns
df.drop(columns=[
    'Timestamp', 'Style', 'Wardrobe_Essentials', 'Fashion_Inspiration', 
    'Shopping_Location', 'Shopping_Timing'
], errors='ignore', inplace=True)

# Handle missing values for categorical columns
categorical_columns = ['Designer_Brands', 'Style_Change', 'Society_Expectations']
for col in categorical_columns:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Apply Label Encoding to categorical columns
label_encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for column in categorical_cols:
    df[column] = label_encoder.fit_transform(df[column])

# Check for missing values after preprocessing
print("Missing values after preprocessing:")
print(df.isnull().sum())

# Separate features and target
X = df.drop("Style_Change", axis=1)
y = df["Style_Change"]

# Split df into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to oversample the minority classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE - X_train_resampled shape:", X_train_resampled.shape)
print("After SMOTE - y_train_resampled shape:", y_train_resampled.shape)

# Check class distribution after SMOTE
print("Class distribution after SMOTE:")
print(y_train_resampled.value_counts())

# Before SMOTE class distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=y_train, palette="viridis")
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()

# After SMOTE class distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=y_train_resampled, palette="viridis")
plt.title("Class Distribution After SMOTE")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()


#Models

# Train SVM
svm_model = SVC(kernel='rbf', random_state=42, class_weight='balanced')
svm_model.fit(X_train_resampled, y_train_resampled)

# Predict
y_pred_svm = svm_model.predict(X_test)

# Evaluate
print("SVM:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
print(classification_report(y_test, y_pred_svm))

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_model.fit(X_train_resampled, y_train_resampled)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluate
print("Decision Tree:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(classification_report(y_test, y_pred_dt))