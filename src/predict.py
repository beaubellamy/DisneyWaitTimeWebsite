import os
import pandas as pd
import joblib

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras import optimizers
# from keras.src.backend.jax.random import dropout

from settings import MODELS_FOLDER, FEATURE_FOLDER


def predict_ride_wait_times(feature_df):

    prediction = pd.DataFrame()

    for ride in feature_df['Ride'].unique():

        ride_label = ride.replace(':', '').replace(' ', '_')
        model_path = os.path.join(MODELS_FOLDER, f'{ride_label}.pkl')
        scalar_path = os.path.join(MODELS_FOLDER, f'{ride_label}_scalar.pkl')

        model = joblib.load(model_path)
        scaler = joblib.load(scalar_path)

        prediction_df = feature_df[feature_df['Ride'] == ride]
        features = prediction_df[['is_weekday', 'Ride_closed', 'Max Temp', 'Avg Temp',
                                  'Yesterday_wait_time', 'Rolling_Avg_7_Days',
                                  'LastWeek_wait_time', 'LastMonth_wait_time',
                                  'Rolling_28D_hr_trend', 'Rolling_28D_3hr_trend',
                                  'Day_sin', 'Day_cos', 'Time_sin', 'Time_cos']]

        scaled_features = scaler.transform(features)
        prediction_df['pred_wait_time'] = model.predict(scaled_features)
        prediction = pd.concat([prediction, prediction_df])

    return prediction

if __name__ == "__main__":

    prediction_date = '20250423'
    features_path = os.path.join(FEATURE_FOLDER, f'features_{prediction_date}.csv')
    features = pd.read_csv(features_path)

    predictions = predict_ride_wait_times(features)
    # output_path = os.path.join(os.path.dirname(__file__), 'predictions.csv')
    predictions.to_csv(f'predictions_{prediction_date}.csv', index=False)

