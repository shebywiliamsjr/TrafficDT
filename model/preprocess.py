import pandas as pd

df = pd.read_csv("../vehicle_tracking_data.csv")

# Define minimum number of frames an ID must exist to be considered valid
FRAME_THRESHOLD = 5

id_counts = df["Vehicle ID"].value_counts()

valid_ids = id_counts[id_counts >= FRAME_THRESHOLD].index
df_filtered = df[df["Vehicle ID"].isin(valid_ids)]

df_filtered.to_csv("filtered_vehicle_tracking_data.csv", index=False)
print(f"Filtered data saved. Removed {len(df) - len(df_filtered)} short-lived IDs.")
