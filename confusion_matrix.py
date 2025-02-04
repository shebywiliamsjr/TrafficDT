from collections import Counter
import numpy as np
import json
from sklearn.metrics import confusion_matrix

json_file_path = "data_for_cm.json"


directions = ['east', 'west', 'north', 'south']

cm = np.zeros((len(directions), len(directions)), dtype=int)

all_actuals = []
all_predicted = []
def get_most_frequent_direction(predictions):
    print(predictions)
    filtered_predictions = [p for p in predictions if p]
    count = Counter(filtered_predictions)
    most_common = count.most_common(1)[0][0]  # Get the most frequent direction
    return most_common

with open(json_file_path, 'r') as f:
    data = json.load(f)

# Iterate over objects in your data and aggregate frames
frame_interval = 50  
for obj_id, frames in data.items():
    frame_ids = list(frames.keys())
    # if obj_id == '2':
    for start_idx in range(0, len(frame_ids), frame_interval):
        interval_frames = frame_ids[start_idx:start_idx + frame_interval]
        actual_predictions = []
        predicted_predictions = []
        
        # Collect actual and predicted directions for the interval
        for frame_id in interval_frames:
            actual = frames[frame_id].get('actual')
            next_pred = frames[frame_id].get('next')
            if actual: actual_predictions.append(actual)
            if next_pred: predicted_predictions.append(next_pred)

        print("Object id", obj_id)
        if len(actual_predictions) > 0 and len(predicted_predictions) > 0:
            most_frequent_actual = get_most_frequent_direction(actual_predictions)
            most_frequent_predicted = get_most_frequent_direction(predicted_predictions)

        print(f"Most frequent prediction", most_frequent_actual, most_frequent_predicted, len(most_frequent_actual), len(most_frequent_predicted))   

        all_actuals.append(most_frequent_actual)
        all_predicted.append(most_frequent_predicted)

        actual_idx = directions.index(most_frequent_actual)
        predicted_idx = directions.index(most_frequent_predicted)
        cm[actual_idx][predicted_idx] += 1

# Print out the confusion matrix
print("Confusion Matrix:")
print(cm)

direction_to_index = {direction: idx for idx, direction in enumerate(directions)}

actual_integers = [direction_to_index[direction] for direction in all_actuals]
predicted_integers = [direction_to_index[direction] for direction in all_predicted]

cm_sklearn = confusion_matrix(actual_integers, predicted_integers)

print("Confusion Matrix (sklearn):")
print(cm_sklearn)