# Preprocessing template 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('train.csv')
X_train = dataset1.iloc[:, [2,4,5,6,7,9,11]].values
y_train = dataset1.iloc[:, 1].values

# Taking care of missing data
# Updated Imputer
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X_train[:, 2:3])
X_train[:, 2:3]=missingvalues.transform(X_train[:, 2:3])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X_train = onehotencoder.fit_transform(X_train).toarray()
# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)
X_train[61][6] = 'Q'
X_train[829][6] = 'S'
labelencoder_X2 = LabelEncoder()
X_train[:, 6] = labelencoder_X2.fit_transform(X_train[:, 6])

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

dataset2 = pd.read_csv('test.csv')
X_test = dataset2.iloc[:, [1,3,4,5,6,8,10]].values

from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X_test[:, :])
X_test[:, :]=missingvalues.transform(X_test[:, :])

labelencoder_X3 = LabelEncoder()
X_test[:, 1] = labelencoder_X3.fit_transform(X_test[:, 1])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X_train = onehotencoder.fit_transform(X_train).toarray()
# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

labelencoder_X4 = LabelEncoder()
X_test[:, 6] = labelencoder_X4.fit_transform(X_test[:, 6])

y_pred = classifier.predict(X_test)
y_final = dataset2.iloc[:, [0]].values
y_pred = y_pred[:,np.newaxis]
y_final = np.append(y_final,y_pred, axis = 1)

np.savetxt("foo.csv", y_final, delimiter="," , fmt = '%.0f')
c = ['PassengerID','Survived']

df = pd.DataFrame(y_final) # A is a numpy 2d array
df.to_csv("A1.csv", header=c,index=False)