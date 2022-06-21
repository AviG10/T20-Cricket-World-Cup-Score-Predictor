# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor


# Load the XgBoost CLassifier model
filename = 'pipe.pkl'
regressor = pickle.load(open(filename, 'rb'))


app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        batting_team = request.form['batting-team']
            
        bowling_team = request.form['bowling-team']
            
        city = request.form['city']

        overs = int(request.form['overs'])
        current_score = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        last_five = int(request.form['runs_in_prev_5'])

        balls_left = 120 - (overs*6)
        crr = current_score/overs


        input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
        my_prediction = int(regressor.predict(input_df)[0])
              
        return render_template('result.html', lower_limit = my_prediction-5, upper_limit = my_prediction+5)



if __name__ == '__main__':
    app.run(debug=True)