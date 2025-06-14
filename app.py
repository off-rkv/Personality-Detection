from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.logger import logging
from sklearn.preprocessing import LabelEncoder
from src.utils import load_object

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/PersonalityPredictor', methods=['GET', 'POST'])
def Personality_Predictor():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Time_spent_Alone = request.form.get("Time_spent_Alone"),
            Stage_fear = request.form.get("Stage_fear"),
            Social_event_attendance = request.form.get("Social_event_attendance"),
            Going_outside = request.form.get("Going_outside"),
            Drained_after_socializing = request.form.get("Drained_after_socializing"),
            Friends_circle_size = request.form.get("Friends_circle_size"),
            Post_frequency = request.form.get("Post_frequency"),
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)

        label_encoder=load_object("artifacts/label_encoder.pkl")
        
        result=label_encoder.inverse_transform(result)

        logging.info('Prediction completed, result: %s', result)
        return render_template('home.html',result=result[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")        

