import pandas as pd
import numpy as np
from collections import Counter

def prepare_lstm_dataset(track_data, output_csv="vehicle_tracking_data.csv"):
    """
    Prepare a CSV dataset from vehicle tracking data for LSTM training.

    :param track_data: Valid vehicle tracking data.
    :param output_csv: Output file name.
    """

    processed_data = []

    for vehicle_id, records in track_data.items():

        entry = records[0][5]
        exit = records[-1][6]

        if (entry is not None and exit is not None) and (entry != exit):
            df = pd.DataFrame(records, columns=["frame", "x_center", "y_center", "speed", "label", "entry_point", "exit_region"])
            df = df.sort_values(by="frame")

            for i in range(0, len(df) - 9, 10):  
                subset = df.iloc[i:i+10]  

                avg_x = subset["x_center"].mean()
                avg_y = subset["y_center"].mean()
                avg_speed = subset["speed"].mean()
                
                # Get the most frequent last_region, ignoring None values
                valid_regions = [region for region in subset["exit_region"] if region is not None]
                
                if valid_regions:
                    last_region = Counter(valid_regions).most_common(1)[0][0]  # Most common region
                else:
                    last_region = "center"

                processed_data.append([vehicle_id, subset["frame"].iloc[-1], avg_x, avg_y, avg_speed, last_region])

    # Convert to DataFrame and save as CSV
    columns = ["Vehicle_ID", "Frame", "Avg_X_Center", "Avg_Y_Center", "Avg_Speed", "Last_Region"]
    df_final = pd.DataFrame(processed_data, columns=columns)
    df_final.to_csv(output_csv, index=False)
