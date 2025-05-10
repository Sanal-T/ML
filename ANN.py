import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('Churn_Modelling.csv')

X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1) 
y = data['Exited']

X.drop(['Geography', 'Gender'], axis=1, inplace=True)

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()

classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=X_train.shape[1]))

classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

classifier.summary()

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f'The accuracy of the model is: {accuracy}')

cl_report = classification_report(y_test, y_pred)
print(cl_report)

print(model_history.history.keys())

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()