import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(df)

del df['id']

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()


print(df)

features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type', 'avg_glucose_level', 'bmi' , 'smoking_status']
label_encoder_map = {}

#convert strings to lowercase
df_copy = df.applymap(lambda s:s.lower() if type(s) == str else s)
#df_copy = df_copy.drop('Survived', axis=1)
#encode only those we're interested in
df_encoded_labels = df_copy.copy()

# create map
for i in range(len(features)):
    labels = df_encoded_labels[features[i]].astype('category').cat.categories.tolist()
    replace_map_comp = {features[i] : {k :v for k, v in zip( labels, list(range(0, len(labels))))}}

    # each replace map is for a column and is added to the total data encoder map
    label_encoder_map.update(replace_map_comp)
    # replace the data with the encoded values according the our map
    df_encoded_labels.replace(replace_map_comp, inplace=True)


print(label_encoder_map)


import pickle as pk
pk.dump(label_encoder_map, open("label_encoder_map.pkl", "wb"))


print(df.isnull().sum())

df['bmi'].fillna(df['bmi'].mean(), inplace=True)
print(df)

print(df.isnull().sum())

df["gender"] = lb_make.fit_transform(df["gender"])
df["ever_married"] = lb_make.fit_transform(df["ever_married"])
df["work_type"] = lb_make.fit_transform(df["work_type"])
df["Residence_type"] = lb_make.fit_transform(df["Residence_type"])
df["smoking_status"] = lb_make.fit_transform(df["smoking_status"])


df

features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type', 'avg_glucose_level', 'bmi' , 'smoking_status']
X = df.loc[:, features]
y = df.loc[:, 'stroke']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto', max_iter=1000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#compute accuracy via sklearn
from sklearn.metrics import accuracy_score
score =accuracy_score(y_test,y_pred)
print("Accuracy: ", score)

import pickle as pk
pk.dump(classifier, open("model.pkl", "wb"))

X_col = df.loc[:, features]
y_col = df.loc[:, 'stroke']

df['Probability of stroke'] = classifier.predict_proba(df[X_col.columns])[:, 1]
print(df[['stroke', 'Probability of stroke']])

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print("Matrix: ", cm)

import seaborn as sns
sns.heatmap(cm, fmt="d", annot=True)

import matplotlib.pyplot as plt
plt.show()