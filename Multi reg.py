import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('homeprices_2.csv')

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())

reg = linear_model.LinearRegression()
X = df.drop('price', axis='columns')
y = df.price
reg.fit(X, y)

prediction = reg.predict([[3000, 3, 40]])
print(f"Prediction for input [3000, 3, 40]: {prediction[0]}")

y_pred = reg.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

area_values = np.linspace(df['area'].min(), df['area'].max(), 100).reshape(-1, 1)
bedrooms_mean = np.mean(df['bedrooms'])
age_mean = np.mean(df['age'])

X_vis = np.hstack((area_values, np.full((100, 1), bedrooms_mean), np.full((100, 1), age_mean)))
y_vis_pred = reg.predict(X_vis)

plt.figure(figsize=(6,3))
plt.scatter(df['area'], df['price'], color='red', label='Actual data')
plt.plot(area_values, y_vis_pred, color='blue', label='Regression line')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("House Price Prediction - Effect of Area")
plt.legend()
plt.show()