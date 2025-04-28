import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('homeprices.csv')

new_df = df.drop('price', axis='columns')
price = df.price

reg = linear_model.LinearRegression()
reg.fit(new_df, price)

y_pred = reg.predict(new_df)

print("Prediction for area=3300:", reg.predict([[3300]]))

mse = mean_squared_error(price, y_pred)
r2 = r2_score(price, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score (Accuracy): {r2}")

plt.figure(figsize=(6, 3))
plt.title("Linear Regression With One Variable")
plt.scatter(new_df, price, label="Data", color="red")
plt.plot(new_df, y_pred, label="Regression Line", color="blue")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.legend()
plt.show()