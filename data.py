# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('data.csv')  # Replace 'data.csv' with your file path

# Step 2: Inspect the data
print(data.head())  # Display the first few rows of the data

# Step 3: Clean the data
# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values
data = data.dropna()

# Step 4: Exploratory Data Analysis (EDA)
# Summary statistics
print(data.describe())

# Step 5: Data Visualization
# Plotting a histogram of a column (e.g., 'age')
plt.hist(data['age'], bins=10, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Step 6: Statistical Analysis
# Correlation matrix
corr_matrix = data.corr()
print(corr_matrix)

# Optional Step 7: Machine Learning (example using linear regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Assuming 'age' is the target variable and 'income' is a feature
X = data[['income']]
y = data['age']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print(predictions)