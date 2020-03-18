# flask.py
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import url_for
import numpy as np
from numpy import loadtxt
from keras.models import load_model
import pandas as pd
from pandas import read_csv
import datetime

app = Flask(__name__)

# load model
model = load_model('Queens_apartment.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST']) # https://github.com/nitinkaushik01/Deploy_Machine_Learning_Model_on_Flask_App/blob/master/Flask_Sample_App/app.py
def predict():
    # grabs the input data when a post request is sent
    ResU= int(request.form['Residential_Units'])
    ComU= int(request.form['Commercial_Units'])
    Gsqft= float(request.form['Gross_sqft'])
    Zip= str(request.form['Zip_Code'])
    Class_C= str(request.form['Class_Category'])
    Year= (request.form['Year_Built'])

    #convert year to the str category that the model will understand
    Year = int(Year)
    if Year < 3:
        Year ='AGE OF BUILDING_1'
    elif Year < 10:
        Year = 'AGE OF BUILDING_2'
    elif Year < 20:
        Year = 'AGE OF BUILDING_3'
    elif Year < 30:
        Year = 'AGE OF BUILDING_4'
    elif Year < 50:
        Year = 'AGE OF BUILDING_5'
    elif Year < 75:
        Year = 'AGE OF BUILDING_6'
    elif Year < 100:
        Year = 'AGE OF BUILDING_7'
    elif Year < 150:
        Year = 'AGE OF BUILDING_8'
    else:
        Year = 'AGE OF BUILDING_9'

    df = pd.read_csv(r'cleaned_data.csv')

    # divides the data set into X (data) and Y (desired predicted value)
    X = df.drop(columns=['SALE PRICE'])
    Y = df['SALE PRICE']

    def predict_price(Residential_Units, Commercial_Units, Gross_sqft, Zip_Code, Class_Category, Year_Built):
        Zip_Code_index= np.where(X.columns==Zip_Code)[0][0] # finds column with the title given in the neighborhood box
        Class_Category_index= np.where(X.columns==Class_Category)[0][0]
        year_index= np.where(X.columns==Year_Built)[0][0]

        x=np.zeros(len(X.columns)) #sets all columns in a data set object x to zero
        x[0]= Residential_Units #changes a specified column from zero to the value assigned to the variable (retreived from the post request)
        x[1]= Commercial_Units
        x[2]= Gross_sqft
        if Zip_Code_index >= 0:
            x[Zip_Code_index] = 1 # assigns a one to the desired neighborhood (one hot encoding)
        if Class_Category_index >= 0:
            x[Class_Category_index] = 1
        if year_index >= 0:
            x[year_index] = 1

        #return model.predict([x])[0]
        test1 = np.array([x])[0] #the x data set object is passed through the ml model
        return model.predict(test1.reshape(1, 88), batch_size=1)

    prediction = predict_price(ResU,ComU,Gsqft,Zip,Class_C,Year) # set up this way to avoid confusion between global and local variables
    return render_template('index.html', prediction_text='Price should be {}'.format(prediction)) # prediction sent to index.html template file

if __name__ == "__main__":
    app.run(debug=True)
