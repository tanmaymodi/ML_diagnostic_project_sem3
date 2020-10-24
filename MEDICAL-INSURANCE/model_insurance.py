import pandas as pd
import pickle

df = pd.read_csv('insurance.csv')

#Mapping sex and smoker to numerical values
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

age_range=[13,25,40,55,80]
slots=['Youth','Adult','Elder Adult','Senior Citizen']
df['Age_bracket']=pd.cut(df['age'],bins=age_range,labels=slots)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
columnsToScale=['bmi','age','children','smoker']
df[columnsToScale]=scaler.fit_transform(df[columnsToScale])

df['region']=df['region'].map({'southeast':1.0,'southwest':0.6,'northeast':0.8,'northwest':0.8})

X=df[['age','sex','bmi','children','smoker','region']]
y=df['charges']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)

pickle.dump(model,open('model_insurance.pkl','wb'))
print("Model Saved")

#INPUT FEATURES
#age
#sex
#bmi
#children
#smoker
#region