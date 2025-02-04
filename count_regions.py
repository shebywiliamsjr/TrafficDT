import json

def count_next_regions(file_path):
    # Initialize counters
    region_counts = {"north": 0, "south": 0, "east": 0, "west": 0}

    # Open and read the file
    with open(file_path, 'r') as file:
        data = json.load(file)  
    # Loop through the data and count occurrences of each next_region
    for key, value in data["54"].items():
        next_region = value.get("next_region")
        if next_region:  # Only count if next_region is not null
            region_counts[next_region] += 1

    return region_counts

file_path = 'data_for_confusion_matrix.json'  
region_counts = count_next_regions(file_path)
print(region_counts)
