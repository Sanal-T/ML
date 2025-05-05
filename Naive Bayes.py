import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv('Employee.csv')
print("Data Head:")
print(data.head())

X = data.drop(columns=['Left_company'])  
Y = data['Left_company']               

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = GaussianNB()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
report = classification_report(Y_test, Y_pred)
print(report)

labels = ['Correct', 'Incorrect']
counts = [accuracy * len(Y_test), (1 - accuracy) * len(Y_test)]

plt.figure(figsize=(6,6))
plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=140)
plt.title('Prediction Accuracy Split')
plt.show()
