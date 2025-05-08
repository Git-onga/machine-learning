import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_excel('housing data.xlsx')

df["Parking"] = df["Parking"].replace({"Yes": 1, "No": 0})
df["Security"] = df["Security"].replace({"Yes": 1, "No": 0})


X = df.drop('Price', axis=1)  # Features
y = df['Price']               # Target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)

# Predicting the target variable for the test set
y_pred = model.predict(x_test)


mae = mean_absolute_error(y_test, y_pred)         # Average absolute error
mse = mean_squared_error(y_test, y_pred)          # Average squared error
rmse = mse ** 0.5                                 # Root mean squared error
r2 = r2_score(y_test, y_pred)                     # R-squared (goodness of fit)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
