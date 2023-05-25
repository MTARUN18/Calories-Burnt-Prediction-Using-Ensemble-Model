#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression


# In[2]:


# Load the user information dataset 
user_data = pd.read_csv('user_info.csv')

# Load the calorie information dataset
calorie_data = pd.read_csv('calories.csv')


# In[3]:


user_data.head()


# In[4]:


calorie_data.head()


# In[5]:


# Merge the datasets based on the User_ID column
data = pd.merge(user_data, calorie_data, on='User_ID')


# In[6]:


data.head()


# In[7]:


num_rows = len(data)

# Get the number of columns using len() on the column index
num_columns = len(data.columns)

print("Number of rows:", num_rows)
print("Number of columns:", num_columns)


# In[8]:


# Drop rows with null values
data = data.dropna()


# In[9]:


sns.set()
plt.figure(figsize=(6,6))
sns.countplot(x=data.Gender)
plt.show()


# In[10]:


# Drop irrelevant columns (e.g., User_ID)
data = data.drop('User_ID', axis=1)


# In[11]:


#outliers
fig,ax = plt.subplots(figsize = (15,10))
sns.boxplot(data=data,width = 0.5,fliersize = 3,ax=ax)


# In[12]:


# Split the dataset into features (X) and target variable (y)
X = data[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
y = data['Calories']


# In[13]:


# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)


# In[14]:


# Calculate the correlation matrix
corr_matrix = data.corr()
# Create a heatmap using the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[15]:


# Plot histogram of Age
plt.hist(data['Age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.show()


# In[16]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# Create the individual models
decision_tree = DecisionTreeRegressor()
random_forest = RandomForestRegressor()
xgb = XGBRegressor()
linear_regression = LinearRegression()
svm_model = SVR()


# In[18]:


# Train the individual models
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
xgb.fit(X_train, y_train)
linear_regression.fit(X_train, y_train)
svm_model.fit(X_train, y_train)



# In[19]:


# Make predictions on the testing data using each model
dt_pred = decision_tree.predict(X_test)
rf_pred = random_forest.predict(X_test)
xgb_pred = xgb.predict(X_test)
lin_pred = linear_regression.predict(X_test)
svm_pred = svm_model.predict(X_test)


# In[20]:


#Decision Tree

# Evaluate the ensemble model
mae = mean_absolute_error(y_test, dt_pred)
mse = mean_squared_error(y_test, dt_pred)
rmse = mean_squared_error(y_test, dt_pred, squared=False)
r2 = r2_score(y_test, dt_pred)

print(f"Decision Tree Mean Absolute Error: {mae}")
print(f"Decision Tree Mean Squared Error: {mse}")
print(f"Decision Tree Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


#Random forest
# Evaluate the ensemble model
mae = mean_absolute_error(y_test, rf_pred)
mse = mean_squared_error(y_test, rf_pred)
rmse = mean_squared_error(y_test, rf_pred, squared=False)
r2 = r2_score(y_test, rf_pred)

print(f"Random forest Mean Absolute Error: {mae}")
print(f"Random forest Mean Squared Error: {mse}")
print(f"Random forest Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


#XGB
# Evaluate the ensemble model
mae = mean_absolute_error(y_test, xgb_pred)
mse = mean_squared_error(y_test, xgb_pred)
rmse = mean_squared_error(y_test, xgb_pred, squared=False)
r2 = r2_score(y_test, xgb_pred)

print(f"XGB Mean Absolute Error: {mae}")
print(f"XGB Mean Squared Error: {mse}")
print(f"XGB Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


#Linear Regression
# Evaluate the ensemble model
mae = mean_absolute_error(y_test, lin_pred)
mse = mean_squared_error(y_test, lin_pred)
rmse = mean_squared_error(y_test, lin_pred, squared=False)
r2 = r2_score(y_test, lin_pred)

print(f"Linear reg  Mean Absolute Error: {mae}")
print(f"Linear reg  Mean Squared Error: {mse}")
print(f"Linear reg  Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


#SVM
# Evaluate the ensemble model
mae = mean_absolute_error(y_test, svm_pred)
mse = mean_squared_error(y_test, svm_pred)
rmse = mean_squared_error(y_test, svm_pred, squared=False)
r2 = r2_score(y_test, svm_pred)

print(f"SVM Mean Absolute Error: {mae}")
print(f"SVM Mean Squared Error: {mse}")
print(f"SVM Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


# In[21]:


#create a stack

stacked_X = np.column_stack((dt_pred,rf_pred,xgb_pred))

# Create the meta-model (linear regression in this example) and train it
meta_model = LinearRegression()
meta_model.fit(stacked_X, y_test)

# Generate the combined predictions using the meta-model
stacked_pred = meta_model.predict(stacked_X)



# In[22]:


# Generate the combined predictions using the meta-model
stacked_pred = meta_model.predict(stacked_X)

# Plot the predictions against the true values
plt.scatter(y_test, stacked_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Plotting the diagonal line for reference
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.show()


# In[23]:


# Evaluate the ensemble model
mae = mean_absolute_error(y_test, stacked_pred)
mse = mean_squared_error(y_test, stacked_pred)
rmse = mean_squared_error(y_test, stacked_pred, squared=False)
r2 = r2_score(y_test, stacked_pred)

print(f"Stacked Model Mean Absolute Error: {mae}")
print(f"Stacked Model Mean Squared Error: {mse}")
print(f"Stacked Model Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


# In[ ]:





# In[ ]:




