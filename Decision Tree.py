import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

colnames = ['Buying_price', 'maint_cost', 'doors', 'persons', 'lug_boot', 'safety', 'decision']

data = pd.read_csv('car_evaluation.csv', names=colnames, header=None)

plt.figure(figsize=(5, 5))
sns.countplot(x='decision', data=data)
plt.title('Count plot for decision')
plt.show()

data.decision.replace('vgood', 'acc', inplace=True)
data.decision.replace('good', 'acc', inplace=True)

print(data['decision'].value_counts())

new_data = data.apply(LabelEncoder().fit_transform)
print(new_data)

X = new_data.drop(['decision'], axis=1)
y = new_data['decision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier(criterion="entropy")

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

cm = confusion_matrix(y_test, dt_pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()

plt.show()