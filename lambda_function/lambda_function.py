import boto3
import pandas as pd
import joblib
import os
from io import BytesIO
from src.predict import predict_ride_wait_times

s3 = boto3.client('s3')
prediction_date = '20250423'
bucket_name = 'your-s3-bucket-name'
features_key = f'features/features_{prediction_date}.csv'
predictions_key = f'predictions/predictions_{prediction_date}.csv'


def lambda_handler(event, context):
    # Download features.csv from S3
    response = s3.get_object(Bucket=bucket_name, Key=features_key)
    feature_df = pd.read_csv(response['Body'])

    # Run predictions
    predictions = predict_ride_wait_times(feature_df)

    # Upload predictions.csv to S3
    csv_buffer = BytesIO()
    predictions.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=predictions_key, Body=csv_buffer.getvalue())

    return {
        'statusCode': 200,
        'body': f'Predictions for {prediction_date} generated and uploaded successfully.'
    }
