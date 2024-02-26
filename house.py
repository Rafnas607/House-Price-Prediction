import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv(r'D:\Data Science\house_env\house_data.csv')


df = df.replace({'?': np.NaN, 'n.a': np.NaN})

df.drop(columns=['date'],axis=1,inplace=True)

df.drop(columns=['long'],axis=1,inplace=True)

df.drop(columns=['lat'],axis=1,inplace=True)


col = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']

df = df[col]

y = df.iloc[:, 4:]
x = df.iloc[:, 0:4]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

lr = LinearRegression()
lr.fit(x_train, y_train)

pickle.dump(lr, open('model.pkl', 'wb'))

