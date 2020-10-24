import pandas as pd
import pickle

dataset=pd.read_excel('dataset.xlsx')

y=dataset['SARS-Cov-2 exam result']
columnsNotToDrop=['SARS-Cov-2 exam result']
g=dataset.columns
datatype=dataset.dtypes
for i in g[6:]:
    ser=dataset[i]
    ser.dropna(inplace=True)
    print(i,'\n',ser.size)
    if ser.size>=500:
        if datatype[i]=='int64' or datatype[i]=='float64':
            columnsNotToDrop.append(i)
            
dataset=dataset[columnsNotToDrop]

dataset=dataset.fillna(dataset.mean())

from sklearn.model_selection import train_test_split
y=dataset['SARS-Cov-2 exam result']
y=y.map({'positive':1,'negative':0})

columnsNotToDrop.pop(0)

X_train, X_test, y_train, y_test = train_test_split(dataset[columnsNotToDrop],y, test_size=0.20, random_state=42)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(class_weight="balanced")
lr.fit(X_train,y_train)
lr_pred=lr.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,lr_pred))

print(classification_report(y_test,lr_pred))

print(confusion_matrix(y_test,lr_pred))

pickle.dump(lr,open('model_covid.pkl','wb'))

#INPUT FEATURES:
#Hematocrit
#Hemoglobin
#Platelets
#Mean platelet volume
#Red blood Cells
#Lymphocytes
#Mean corpuscular hemoglobin concentration (MCHC)
#Leukocytes
#Basophils
#Mean corpuscular hemoglobin (MCH)
#Eosinophils
#Mean corpuscular volume (MCV)
#Monocytes
#Red blood cell distribution width (RDW)
#Neutrophils
#Proteina C reativa mg/dL