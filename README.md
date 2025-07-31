Overview:
Developed a Python-based intelligent analytics system to process, classify, and visualize unstructured enterprise log data. The system identifies critical errors, assigns priority levels and flags, clusters similar log messages using NLP and unsupervised learning, and provides time-based visual insights into high-priority system failures.

Key Features:

🔎 Error Classification: Applied regex-based patterns to classify logs into error types (e.g., TimeoutError, ResourceFailure).

⚠️ Priority & Flag Assignment: Automatically labeled logs with priority levels (high/medium/low), binary error flags, and manual intervention indicators.

🧠 Machine Learning Model: Built a logistic regression model with a pipeline using TF-IDF, OneHotEncoder, and StandardScaler to predict error flags.

📊 Clustering & Visualization: Used KMeans clustering and PCA to group similar logs and visualize them in 2D space.

📅 Time-based Analysis: Generated day-wise, week-wise, and month-wise bar charts of high-priority errors using Seaborn and Matplotlib.

📁 Tools Used: Python, Pandas, Scikit-learn, Seaborn, Matplotlib, Regex, TF-IDF, KMeans, Logistic Regression.

Outcome:
Enabled real-time analysis of enterprise logs, helping engineering teams detect and prioritize critical system issues efficiently over time.



