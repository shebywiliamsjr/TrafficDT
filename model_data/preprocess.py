import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


# TODO: Remove the vehicle information that has < SEQUQNCE_LEN records (Not enough data for reading sequence)


dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(dir, "preprocessed_data.csv")

df = pd.read_csv("./vehicle_tracking_data.csv")

# Define minimum number of frames an ID must exist to be considered valid
FRAME_THRESHOLD = 5

# Define minumum speed that a vehicle must have to be in green light 
SPEED_THRESHOLD = 4

id_counts = df["Vehicle_ID"].value_counts()

valid_ids = id_counts[id_counts >= FRAME_THRESHOLD].index
df_filtered = df[df["Vehicle_ID"].isin(valid_ids)]

df = df_filtered.sort_values(by=["Vehicle_ID", "Frame"])

df["is_red_light"] = 0

for vehicle_id, group in df.groupby("Vehicle_ID"):
    stop = 0
    prev_region = None

    for id in group.index:
        speed = df.at[id, "Avg_Speed"]
        region = df.at[id, "Last_Region"]

        if speed <= SPEED_THRESHOLD:
            if prev_region is None or prev_region == region:
                stop += 1
            else: 
                stop = 0
        else:
            stop = 0
        
        if stop >= FRAME_THRESHOLD:
            df.at[id, "is_red_light"] = 1
        
        prev_region = region

label_encoder = LabelEncoder()
df["Region_Encoded"] = label_encoder.fit_transform(df["Last_Region"])

# df_filtered.to_csv("filtered_vehicle_tracking_data.csv", index=False)
# print(f"Filtered data saved. Removed {len(df) - len(df_filtered)} short-lived IDs.")

df.to_csv(file_path, index=False)

