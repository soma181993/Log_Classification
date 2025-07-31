

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------- STEP 1: LOAD & CLEAN DATA --------------------
data = pd.read_csv('enterprise_system_unstructured_data.csv')
data = data.dropna()

# -------------------- STEP 2: CLASSIFY ERROR MESSAGE --------------------
def classify_error_message(message):
    patterns = {
        'InvalidData': r'(?i)invalid data detected',
        'ResourceFailure': r'(?i)failed to allocate resources',
        'TimeoutError': r'(?i)timeout occurred',
        'ProcessingError': r'(?i)error encountered during process',
        'ManualIntervention': r'(?i)manual intervention required'
    }
    for error_type, pattern in patterns.items():
        if re.search(pattern, message):
            return error_type
    return 'Unknown'

data['issue_type'] = data['message_text'].apply(classify_error_message)

# -------------------- STEP 3: ASSIGN PRIORITY & FLAGS --------------------
def assign_priority_and_flags(issue_type):
    high_issues = ['ResourceFailure', 'TimeoutError', 'ManualIntervention']
    if issue_type in high_issues:
        return {'priority_level': 'high', 'error_flag': 1, 'manual_intervention': 1 if issue_type == 'ManualIntervention' else 0}
    elif issue_type == 'InvalidData':
        return {'priority_level': 'medium', 'error_flag': 1, 'manual_intervention': 0}
    else:
        return {'priority_level': 'low', 'error_flag': 0, 'manual_intervention': 0}

assigned = data['issue_type'].apply(assign_priority_and_flags)
assigned_df = pd.DataFrame(list(assigned))
data = pd.concat([data.drop(columns=['priority_level', 'error_flag', 'manual_intervention'], errors='ignore'), assigned_df], axis=1)

# Encode priority level
data['priority_level_encoded'] = data['priority_level'].apply(lambda x: 1 if str(x).lower() == 'high' else 0)
data['manual_intervention'] = data['manual_intervention'].astype(int)

# -------------------- STEP 4: TRAIN MODEL --------------------
X = data[['source_system', 'message_text', 'process_type', 'user_action', 'resource_consumption', 'response_time', 'priority_level_encoded']]
y = data['error_flag']  # Change to 'manual_intervention' or 'priority_level_encoded' as needed

text_transformer = TfidfVectorizer(max_features=100)
cat_transformer = OneHotEncoder(handle_unknown='ignore')
num_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'message_text'),
        ('cat', cat_transformer, ['source_system', 'process_type', 'user_action']),
        ('num', num_transformer, ['resource_consumption', 'response_time', 'priority_level_encoded'])
    ]
)

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------- STEP 5: CLUSTER LOGS --------------------
logs = data['message_text'].tolist()
tfidf = TfidfVectorizer(stop_words='english')
X_logs = tfidf.fit_transform(logs)

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_logs)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_logs.toarray())

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='rainbow')
plt.title("Log Message Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print clustered logs
for i in range(4):
    print(f"\nCluster {i}:")
    for log, label in zip(logs, labels):
        if label == i:
            print(" -", log)

# -------------------- STEP 6: PLOT HIGH PRIORITY ERRORS --------------------
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
high_priority_errors = data[
    (data['error_flag'] == 1) &
    (data['priority_level'].str.lower() == 'high')
].copy()

# Add time features
high_priority_errors['day'] = high_priority_errors['timestamp'].dt.date
high_priority_errors['week'] = high_priority_errors['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
high_priority_errors['month'] = high_priority_errors['timestamp'].dt.to_period('M').dt.to_timestamp()

# Plotting
sns.set(style="whitegrid")

# Day-wise
plt.figure(figsize=(12, 5))
sns.countplot(data=high_priority_errors, x='day')
plt.title("High Priority Errors per Day")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Week-wise
plt.figure(figsize=(12, 5))
sns.countplot(data=high_priority_errors, x='week')
plt.title("High Priority Errors per Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Month-wise
plt.figure(figsize=(12, 5))
sns.countplot(data=high_priority_errors, x='month')
plt.title("High Priority Errors per Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()