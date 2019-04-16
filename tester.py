import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
classifier = pickle.load(open('classifier', 'rb'))
dataset = pd.read_csv('bank-additional-full.csv', sep=';')
X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, 20].values
dataset.describe()
dataset.head()


# Encoding categorical data
labelencoder_y_1 = LabelEncoder()
y[:] = labelencoder_y_1.fit_transform(y[:])
col = ColumnTransformer([("oh_enc", OneHotEncoder(categories=[['admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'],
                                                                 ['divorced','married','single','unknown'],
                                                                 ['basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'],
                                                                 ['no','yes','unknown'],
                                                                 ['no','yes','unknown'],
                                                                 ['no','yes','unknown'],
                                                                 ['cellular','telephone'],
                                                                 ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec'],
                                                                 ['mon','tue','wed','thu','fri'],
                                                                 ['failure','nonexistent','success']]), [1, 2, 3, 4, 5, 6, 7, 8, 9, 14]), ], remainder = 'passthrough')

X = col.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_ts = np.zeros(len(y_test), int)
y_tr = np.zeros(len(y_train), int)
for i in range(len(y_test)):
    if y_test[i] == 1:
        y_ts[i] = 1
    else:
        y_ts[i] = 0
print(y_ts)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_ts, y_pred)
print(cm)

