import pandas as pd
import pickle
#reading the csv file(dataset)
df=pd.read_csv('dataset_chronic_kidney_disease.csv')

#Getting input features from the dataset by dropping the output class
X=df.drop(columns={'Class'})
#Getting the output
y=df['Class']

from sklearn.ensemble import RandomForestClassifier
#Creating an object for our RandomForestClassifier model with the name rfc
rfc=RandomForestClassifier()
#Fitting the model
rfc.fit(X,y)
#dump the model object
pickle.dump(rfc,open('model.pkl','wb'))
#returns the pickled representation of the object