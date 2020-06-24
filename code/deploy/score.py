import json
import logging
import os

import joblib
import pandas as pd

# from azureml.monitoring import ModelDataCollector
from sktime.utils.load_data import from_long_to_nested

from utils import create_response, get_connection_device_id

TIMESERIESLENGTH = 10


def init():
    global model
    # global inputs_dc, prediction_dc
    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = "model.pkl"
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], model_filename)
    model = joblib.load(model_path)

    logging.info("Model loaded.")

    # inputs_dc = ModelDataCollector(
    #     "sample-model",
    #     designation="inputs",
    #     feature_names=["feat1", "feat2", "feat3", "feat4"],
    # )
    # prediction_dc = ModelDataCollector(
    #     "sample-model", designation="predictions", feature_names=["prediction"]
    # )


def run(data):
    has_error = False

    logging.info("started run.")

    # CONVERT STREAM ANALYTICS TO SKTIME FORMAT
    logging.info("loading json.")
    data = json.loads(data)
    logging.info("json loaded.")

    # Parse timestamps and temperature data
    time_created_start = data.get("allevents")[0].get("timeCreated")
    time_created_end = data.get("allevents")[-1].get("timeCreated")
    temperature_data = [event.get("temperature") for event in data.get("allevents")]

    logging.info(f"time_created_start: {time_created_start}")
    logging.info(f"time_created_end: {time_created_end}")
    logging.info(f"temperature_data: {temperature_data}")

    # Check connection_device_id
    connection_device_id, has_error, error_message = get_connection_device_id(data)
    if has_error:
        return create_response(has_error=has_error, error_message=error_message)

    # Assert time series has at least TIMESERIESLENGTH elements
    if len(temperature_data) < TIMESERIESLENGTH:
        error_message = f"Time series of length {len(temperature_data)} does not have enough samples ({TIMESERIESLENGTH} samples required)."
        logging.warning(error_message)
        return create_response(has_error=True, error_message=error_message)

    # Convert data to sktime format
    case_id, dim_id = 0, 0
    try:
        long_data = [
            [case_id, dim_id, reading_id, reading_data]
            for reading_id, reading_data in enumerate(
                temperature_data[-TIMESERIESLENGTH:]
            )
        ]
    except Exception as e:
        error_message = (
            f"Could not convert dataset to long format due to exception: '{e}'"
        )
        logging.error(error_message)
        return create_response(has_error=True, error_message=error_message)

    # Predict
    long_df = pd.DataFrame(
        long_data, columns=["case_id", "dim_id", "reading_id", "value"]
    )
    sktime_df = from_long_to_nested(long_df)
    prediction = model.predict(sktime_df).tolist()[0]

    # inputs_dc.collect(sktime_df)  # this call is saving our input data into Azure Blob
    # prediction_dc.collect(
    #     prediction
    # )  # this call is saving our input data into Azure Blob

    return create_response(
        prediction=prediction,
        connection_device_id=connection_device_id,
        time_created_start=time_created_start,
        time_created_end=time_created_end,
    )
