import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
df = pd.read_csv('https://raw.githubusercontent.com/Ncalo19/Queens_price_predictor/master/data_pandas/rollingsales_queens.csv')
df = df.drop(columns=['BOROUGH', 'APARTMENT NUMBER', 'BLOCK', 'LOT', 'BUILDING CLASS AT PRESENT', 'EASE-MENT', 'BOROUGH', 'TAX CLASS AT PRESENT', 'BLOCK', 'LOT', 'BUILDING CLASS AT PRESENT', 'ADDRESS', 'NEIGHBORHOOD', 'SALE DATE', 'LAND SQUARE FEET', 'TOTAL UNITS'])
df = df.rename(columns={' SALE PRICE ': 'SALE PRICE'})
df.dropna
df['SALE PRICE']= pd.to_numeric(df['SALE PRICE'], errors = 'coerce')
df['GROSS SQUARE FEET']= pd.to_numeric(df['GROSS SQUARE FEET'], errors = 'coerce')
df['RESIDENTIAL UNITS']= pd.to_numeric(df['RESIDENTIAL UNITS'], errors = 'coerce')
df['COMMERCIAL UNITS']= pd.to_numeric(df['COMMERCIAL UNITS'], errors = 'coerce')
df['YEAR BUILT']= pd.to_numeric(df['YEAR BUILT'], errors = 'coerce')
x = df[df['SALE PRICE'] > 8000000].index
df = df.drop(x, inplace = False)
x = df[df['SALE PRICE'] < 25000].index
df = df.drop(x, inplace = False)
df = df.drop_duplicates(df.columns)
df = df.drop(columns=['BUILDING CLASS AT TIME OF SALE', 'TAX CLASS AT TIME OF SALE'])
df = df[df["GROSS SQUARE FEET"] < df["GROSS SQUARE FEET"].quantile(0.95)]
df = df[df["RESIDENTIAL UNITS"] < df["RESIDENTIAL UNITS"].quantile(0.95)]
df = df[df["COMMERCIAL UNITS"] < df["COMMERCIAL UNITS"].quantile(0.95)]
df = pd.get_dummies(df, columns=['BUILDING CLASS CATEGORY', 'ZIP CODE'])
df['YEAR BUILT'] = (datetime.datetime.now().year - df['YEAR BUILT'])
df = df.rename(columns = {'YEAR BUILT' : 'AGE OF BUILDING'})
df_SALEPRICE = df.pop('SALE PRICE') #move sale price to the last column
df['SALE PRICE']=df_SALEPRICE
bins = [0,3,10,20,30,50,75,100,150,1000]
labels = [1,2,3,4,5,6,7,8,9]
df['AGE OF BUILDING'] = pd.cut(df['AGE OF BUILDING'], bins=bins, labels=labels, right=True)
df = pd.get_dummies(df, columns=['AGE OF BUILDING'])
df.columns = [col.replace('BUILDING CLASS CATEGORY_', '') for col in df.columns]
df.columns = [col.replace('ZIP CODE_', '') for col in df.columns]
df.columns = [col.replace('.0', '') for col in df.columns]

df.to_csv(r'cleaned_data.csv', index=False)
