import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
 
# Load dataset
df = pd.read_csv('po2_data.csv')
 
# Check for NA values
na_values = df.isna().sum()
 
# Print the count of NA values for each column
print(na_values)

#Task 1
#Building basic Linear Regression Model
 
#UPDRS(MOTOR_UPDRS)
 
# Selecting features and target variables for prediction
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
y = df[['motor_updrs']]
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
 
#Total UPDRS
 
# Selecting features and target variables for prediction
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
y = df[['total_updrs']]
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
 
#Task 2
#Analysing LR Model By Spliting the data in training model set and testing model set
 
#50-50
 
# Selecting features and target variables for prediction
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
y = df[['motor_updrs']]
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
 
# Selecting features and target variables for prediction
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
y = df[['total_updrs']]
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}') 
 
#70-30
 
# Selecting features and target variables for prediction
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
y = df[['motor_updrs']]
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
 
# Selecting features and target variables for prediction
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
y = df[['total_updrs']]
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
 
#80-20
 
# Selecting features and target variables for prediction
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
y = df[['motor_updrs']]
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
 
# Selecting features and target variables for prediction
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
y = df[['total_updrs']]
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
 
#Task 3
#Running log-transform and collinearity analysis and Rebuling LR model according to it and testing it
 
# Applying log-transform to numeric features (excluding target variables and identifier)
features = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#']).select_dtypes(include=[np.number]).columns
df[features] = df[features].apply(np.log1p)  # Using log1p to also handle 0 values
 
# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
 
# Drop or impute NaN values - here, we are dropping for simplicity
df.dropna(inplace=True)
 
# Getting features
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
 
# Calculating VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
 
# Sorting the VIF values in ascending order
vif_data = vif_data.sort_values(by='VIF', ascending=False)
 
print(vif_data)
 
# Selecting features and target variables for prediction after transformations
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#','sex'])
y = df[['motor_updrs']]
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
 
# Selecting features and target variables for prediction after transformations
X = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#','sex'])
y = df[['total_updrs']]
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
 
#Task 4
#Running standardisation and Gaussian Transformation and Rebuling LR model according to it and testing it
 
# Separate features and targets
features = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
targets = df[['motor_updrs']]
 
# Standardize the features
scaler = StandardScaler()
standardized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
 
# Apply Yeo-Johnson transformation
transformer = PowerTransformer(method='yeo-johnson')
gaussian_features = pd.DataFrame(transformer.fit_transform(standardized_features), columns=standardized_features.columns)
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(gaussian_features, targets, test_size=0.4, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
 
# Separate features and targets
features = df.drop(columns=['motor_updrs', 'total_updrs', 'subject#'])
targets = df[['total_updrs']]
 
# Standardize the features
scaler = StandardScaler()
standardized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
 
# Apply Yeo-Johnson transformation
transformer = PowerTransformer(method='yeo-johnson')
gaussian_features = pd.DataFrame(transformer.fit_transform(standardized_features), columns=standardized_features.columns)
 
# Splitting data into training and testing sets (60-40)
X_train, X_test, y_train, y_test = train_test_split(gaussian_features, targets, test_size=0.4, random_state=42)
 
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predicting on the test set
predictions = model.predict(X_test)
 
# Calculate MAE, MSE and r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
 
# Assuming AMSE and NAM as custom metrics
amse = mse  
nam = mae   
 
# Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
 
# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'AMSE: {amse}')  
print(f'NAM: {nam}')    
print(f'r^2: {r2}')
print(f'adjusted-r^2: {adjusted_r2}')
