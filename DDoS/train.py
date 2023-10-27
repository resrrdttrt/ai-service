import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# Step 1: Load the dataset
data = pd.read_csv('APA-DDoS-Dataset.csv')

# Step 2: Data Cleaning and Preprocessing
# Drop any rows with missing values
data = data.dropna()
# print(data['label'].unique)
# Convert non-numeric columns to numeric using LabelEncoder
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Drop the 'time' column
data = data.drop('frame_time', axis=1)
data = data.drop('ip_dst', axis=1)
data = data.drop('ip_src', axis=1)


# Drop any other columns that are not suitable for KNN classification (if needed)

# Step 3: Data Reduction (if needed)
# Sample a subset of the data (optional) to reduce computational time
# data = data.sample(n=10000)  # Example: Randomly sample 10,000 rows from the dataset

# Step 4: K-Nearest Neighbors (KNN) Classification
# Separate features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
k_value = 5  # You can set the desired value for k here
knn_classifier = KNeighborsClassifier(n_neighbors=k_value)

# Train the classifier
knn_classifier.fit(X_train, y_train)

import joblib
filename = 'ddos_knn_classifier_model.pkl'
joblib.dump(knn_classifier, filename)
print(f"Model saved as {filename}")
