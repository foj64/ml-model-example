import numpy as np
import pandas as pd

training_df = pd.read_csv('storepurchasedata.csv')

training_df.describe()

x = training_df.iloc[:, :-1].values
y = training_df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors= 5, metric='minkowski', p=2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

new_pred = classifier.predict(sc.transform(np.array([[40,80000]])))

new_prob = classifier.predict_proba(sc.transform(np.array([[40,80000]])))[:,1]

import pickle

model_file = "classifier.pickle"
pickle.dump(classifier, open(model_file, 'wb'))

scaler_file = "sc.pickle"
pickle.dump(sc, open(scaler_file, 'wb'))


